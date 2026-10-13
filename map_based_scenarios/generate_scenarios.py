"""Road-constrained map environments on a fixed OSM map.

Environments share the real map, the three-cell deployment, the road mobility
rules and the radio/measurement mechanism. They differ only by a handful of
sampled conditions plus the random seed, with no steady/tide/decay/surge
profiles and no prescribed head-count curve.

Time is split into two layers:

- 1 s physics: movement, entering, exit intent, leaving and connectivity.
- 10 s snapshots: one observation and one control step per snapshot.

An environment keeps 101 snapshots (0, 10, ..., 1000 s) and therefore executes
100 control steps. Only snapshot positions are stored in the scenario file; the
per-second trajectory is internal to generation and is never given to the agent.

Run from the repository root:
    python map_based_scenarios/generate_scenarios.py --num-scenarios 10 --output-dir map_based_scenarios/scenarios
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml

try:
    import osmnx as ox
except ImportError:
    ox = None

# Candidate values for the sampled environment conditions. They are simulation
# settings, not calibrated statistics; override them with --config.
DEFAULT_CONDITION_CANDIDATES = {
    "initial_users": [30, 45, 60],
    "arrival_rate_per_second": [0.01, 0.02, 0.03],
    "mean_exit_intent_seconds": [600.0, 900.0, 1200.0],
    "speed_mps_range": [[1.0, 1.4], [1.2, 1.6], [1.4, 1.8]],
    "west_entry_probability": [0.25, 0.5, 0.75],
    "mean_connected_seconds": [300.0, 600.0, 900.0],
}
CONDITION_ORDER = tuple(DEFAULT_CONDITION_CANDIDATES)


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


MAP_PATH = Path(__file__).resolve().with_name("sdnu.osm")
DEFAULT_BASE_STATIONS = [
    {"id": 1, "position": [116.8239735, 36.5429144], "tx_power": 46},
    {"id": 2, "position": [116.8274521, 36.5456314], "tx_power": 46},
    {"id": 3, "position": [116.8306442, 36.5443304], "tx_power": 46},
]
# Coordinates are configuration assumptions, not surveyed gate boundaries.
DEFAULT_GATES = {
    "West_Gate": {"lat": 36.543179, "lon": 116.822931, "radius_m": 0.0},
    "North_Gate": {"lat": 36.551051, "lon": 116.828600, "radius_m": 60.0},
}


def child_rng(seed, *labels):
    raw = json.dumps([int(seed), *labels], separators=(",", ":")).encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(raw).digest()[:8], "little"))


def sample_environment_conditions(num_environments, seed, candidates=None, overrides=None):
    """Sample one condition set per environment and record the seeds used.

    Conditions are drawn from candidate lists rather than sampled from
    continuous ranges, so every environment is reproducible and its difference
    from the others is explainable by the recorded values. Each parameter is
    cycled through its candidates before repeating, so a small pool still covers
    every candidate value instead of relying on luck.

    Returns a list of condition dictionaries; each carries the environment seed,
    the conditions seed and a registry of the sampled values.
    """
    pool = copy.deepcopy(DEFAULT_CONDITION_CANDIDATES)
    for name, values in (candidates or {}).items():
        if name not in pool:
            raise ValueError("Unknown condition candidate: %s" % name)
        if not values:
            raise ValueError("Candidate list for %s must not be empty" % name)
        pool[name] = copy.deepcopy(list(values))
    for name, value in (overrides or {}).items():
        if name not in pool:
            raise ValueError("Unknown condition override: %s" % name)
        pool[name] = [copy.deepcopy(value)]

    plan_rng = child_rng(seed, "condition_plan")
    # Balanced coverage without lockstep. Environments are walked in blocks the
    # size of the candidate count, every parameter is reshuffled inside each
    # block, and each parameter starts at its own offset so the parameters are
    # not tied to one another. Every candidate value is therefore used equally
    # often while ten environments still differ from one another, which a plain
    # cycle cannot do (lockstep cycling repeats the same handful of value tuples).
    phases = {name: int(plan_rng.integers(len(values))) for name, values in pool.items()}
    orders = {name: list(range(len(values))) for name, values in pool.items()}
    filled = {name: 0 for name in pool}

    conditions = []
    for index in range(num_environments):
        environment_seed = int(child_rng(seed, "environment", index).integers(0, 2**32))
        conditions_seed = int(child_rng(seed, "conditions", index).integers(0, 2**32))
        rng = np.random.default_rng(conditions_seed)
        sampled = {}
        registry = {}
        for name, values in pool.items():
            count = len(values)
            if filled[name] == 0:
                plan_rng.shuffle(orders[name])
            position = (orders[name][filled[name]] + phases[name]) % count
            filled[name] = (filled[name] + 1) % count
            sampled[name] = copy.deepcopy(values[position])
            registry[name] = copy.deepcopy(values)
        condition = dict(sampled)
        condition["index"] = index
        condition["seed"] = environment_seed
        condition["conditions_seed"] = conditions_seed
        condition["condition_registry"] = registry
        condition["sampled_at_rng_state"] = json.loads(json.dumps(rng.bit_generator.state, default=str))
        conditions.append(condition)
    return conditions


class OSMScenarioGenerator:
    """Road-constrained random walks with independent traffic and connectivity."""

    def __init__(self, osm_file_path=MAP_PATH, gates=None):
        self.osm_file_path = str(Path(osm_file_path).resolve())
        self.gates_config = copy.deepcopy(gates if gates is not None else DEFAULT_GATES)
        self.osm_graph = None
        self.node_positions_cache = {}
        self.gate_nodes_by_name = {}
        self._paths = {}
        self._walk_out_cache = {}

    def load_osm_data(self):
        if self.osm_graph is not None:
            return
        if ox is None:
            raise ImportError("Map generation requires osmnx and networkx in the project environment")
        import networkx as nx

        # A single pedestrian network; no change of mobility model by profile.
        graph = ox.graph_from_xml(self.osm_file_path).to_undirected()
        component = max(nx.connected_components(graph), key=len)
        self.osm_graph = graph.subgraph(component).copy()
        graph = self.osm_graph
        xs = [float(d["x"]) for _, d in graph.nodes(data=True)]
        ys = [float(d["y"]) for _, d in graph.nodes(data=True)]
        self.origin = (min(xs), min(ys), (min(ys) + max(ys)) / 2)
        self.node_positions_cache = {
            int(node): self._project(float(d["x"]), float(d["y"]))
            for node, d in graph.nodes(data=True)
        }
        for u, v, key, data in graph.edges(keys=True, data=True):
            geometry = data.get("geometry")
            points = ([self._project(float(x), float(y)) for x, y in geometry.coords]
                      if geometry is not None else [self.node_positions_cache[u], self.node_positions_cache[v]])
            # Store orientation and metric length of each edge, including bends.
            if distance(points[-1], self.node_positions_cache[u]) < distance(points[0], self.node_positions_cache[u]):
                points.reverse()
            data["_points"] = [self.node_positions_cache[u], *points, self.node_positions_cache[v]]
            data["_start"] = u
            data["_route_length"] = sum(distance(a, b) for a, b in zip(data["_points"], data["_points"][1:]))
        if not self.gates_config:
            raise ValueError("At least one configured map gate is required")
        for name, gate in self.gates_config.items():
            point = self._project(float(gate["lon"]), float(gate["lat"]))
            ranked = sorted(self.node_positions_cache, key=lambda n: (distance(point, self.node_positions_cache[n]), n))
            radius = float(gate.get("radius_m", 0))
            if radius < 0 or distance(point, self.node_positions_cache[ranked[0]]) > float(gate.get("max_snap_distance_m", 150)):
                raise ValueError("Gate %s is too far from the connected road network" % name)
            nodes = [n for n in ranked if distance(point, self.node_positions_cache[n]) <= radius]
            self.gate_nodes_by_name[name] = nodes or ranked[:1]
        self.gate_nodes = {n for nodes in self.gate_nodes_by_name.values() for n in nodes}
        self.interior_nodes = sorted(set(graph.nodes()) - self.gate_nodes)
        if not self.interior_nodes:
            raise ValueError("The road network must have interior destinations")

    def _project(self, lon, lat):
        min_lon, min_lat, center_lat = self.origin
        return (
            (lon - min_lon) * math.pi / 180 * 6371000 * math.cos(math.radians(center_lat)),
            (lat - min_lat) * math.pi / 180 * 6371000,
        )

    def _get_scene_bounds_meters(self):
        self.load_osm_data()
        points = list(self.node_positions_cache.values())
        return 0.0, 0.0, max(p[0] for p in points), max(p[1] for p in points)

    def _convert_base_stations_to_meters(self, stations):
        self.load_osm_data()
        result = copy.deepcopy(stations)
        for station in result:
            station["position"] = list(self._project(*station["position"]))
        return result

    def _route(self, source, target):
        import networkx as nx

        key = (source, target)
        if key not in self._paths:
            nodes = nx.shortest_path(self.osm_graph, source, target, weight="_route_length")
            points = [self.node_positions_cache[source]]
            for u, v in zip(nodes, nodes[1:]):
                edges = self.osm_graph.get_edge_data(u, v)
                edge_key = min(edges, key=lambda k: (edges[k]["_route_length"], k))
                edge = edges[edge_key]
                segment = edge["_points"]
                if edge["_start"] != u:
                    segment = list(reversed(segment))
                for point in segment:
                    if distance(points[-1], point) > 1e-9:
                        points.append(point)
            self._paths[key] = points
        return self._paths[key]

    @staticmethod
    def _advance(user, distance):
        points = user["route"]
        while user["route_index"] < len(points) and distance > 0:
            target = points[user["route_index"]]
            remaining = math.hypot(user["position"][0] - target[0], user["position"][1] - target[1])
            if remaining <= distance:
                user["position"] = target
                user["route_index"] += 1
                distance -= remaining
            else:
                ratio = distance / remaining
                user["position"] = tuple(a + ratio * (b - a) for a, b in zip(user["position"], target))
                break
        return user["route_index"] >= len(points)

    def _choose_walk_edge(self, user):
        node = user["node"]
        choices = [(neighbor, key) for neighbor in sorted(self.osm_graph[node])
                   for key in sorted(self.osm_graph[node][neighbor])
                   if self.osm_graph[node][neighbor][key]["_route_length"] > 1e-8]
        onward = [edge for edge in choices if edge[0] != user.get("previous_node")]
        choices = onward or choices
        if not choices:
            raise ValueError("Road node %s has no traversable edge" % node)
        neighbor, key = choices[int(user["walk_rng"].integers(len(choices)))]
        edge = self.osm_graph[node][neighbor][key]
        points = edge["_points"]
        if edge["_start"] != node:
            points = list(reversed(points))
        user.update(route=points, route_index=1, edge_start=node, edge_end=neighbor)

    def _advance_walk(self, user, budget):
        # Consume the entire time step even when crossing multiple junctions.
        while budget > 1e-8:
            points = [user["position"], *user["route"][user["route_index"]:]]
            remaining = sum(distance(a, b) for a, b in zip(points, points[1:]))
            arrived = self._advance(user, budget)
            budget -= min(budget, remaining)
            if not arrived:
                break
            user["previous_node"], user["node"] = user["edge_start"], user["edge_end"]
            self._choose_walk_edge(user)

    def _start_departure(self, user):
        # On an edge, compare routes through both endpoints without teleporting
        # or cutting across bends in the road geometry.
        index, route = user["route_index"], user["route"]
        forward = [user["position"], *route[index:]]
        backward = [user["position"], *reversed(route[:index])]
        candidates = []
        for prefix, node in ((forward, user["edge_end"]), (backward, user["edge_start"])):
            # Nearest reachable gate from each endpoint; the closer endpoint wins.
            exit_node, exit_gate, _ = self._nearest_gate_node(node)
            candidate = prefix + self._route(node, exit_node)[1:]
            length = sum(distance(a, b) for a, b in zip(candidate, candidate[1:]))
            candidates.append((length, candidate, exit_node, exit_gate))
        length, route_points, exit_node, exit_gate = min(candidates, key=lambda item: item[0])
        user.update(state="departing", route=route_points, route_index=1,
                    exit_node=exit_node, exit_gate=exit_gate)
        if user.get("record") is not None:
            user["record"]["exit_gate"] = exit_gate
            user["record"]["exit_node"] = exit_node

    @staticmethod
    def _disconnect_intervals(seed, record, simulation_seconds, snapshot_seconds, connection):
        """Return connectivity gaps as half-open snapshot-index windows.

        The environment builds them by slicing the snapshot-length visibility
        array, so the boundaries are snapshot indices: a user is invisible over
        [start_step, end_step). The physical process runs on the 1 s clock, so the
        gap length is generated in seconds and then projected onto the snapshot
        grid.
        """
        mean_online = float(connection["mean_connected_seconds"])
        if mean_online == 0:  # Explicit opt-out for controlled comparisons.
            return []
        offline = connection["disconnected_seconds_range"]
        rng = child_rng(seed, "connection", record["ue_id"])
        probability = -math.expm1(-simulation_seconds / mean_online)
        first, last = record["spawn_step"], record["leave_step"]
        intervals = []
        cursor = first * snapshot_seconds
        end = last * snapshot_seconds
        while cursor < end:
            start = cursor + int(rng.geometric(probability)) * simulation_seconds
            if start >= end:
                break
            duration = float(rng.uniform(*offline))
            cursor = min(end, start + max(simulation_seconds, duration))
            start_index = math.ceil(start / snapshot_seconds - 1e-9)
            end_index = min(last, math.floor(cursor / snapshot_seconds + 1e-9))
            if end_index > start_index:
                # Keys match the lifecycle schema consumed by
                # scenario_lifecycle.split_disconnected_lifecycles, which slices
                # the snapshot array over [start_step, end_step).
                intervals.append({"start_step": start_index, "end_step": end_index})
        return intervals

    def _build_scenario_config(self, base_stations, ue_trajectories, total_steps, scenario_name):
        """Preserve the existing constant-cohort adapter for older callers/tests."""
        _, _, width, height = self._get_scene_bounds_meters()
        lifecycles = [
            {
                "ue_id": int(device["id"]), "spawn_step": 0, "leave_step": int(total_steps),
                "initial_base_station_id": int(device["base_station_id"]),
                "trajectory": device["trajectory"], "disconnect_intervals": [],
            }
            for device in sorted(ue_trajectories, key=lambda d: int(d["id"]))
        ]
        count = len(lifecycles)
        return {
            "schema_version": "ue_lifecycle_v1",
            "scene": {"width": width, "height": height, "total_steps": total_steps},
            "base_stations": base_stations, "max_active_users": count, "total_unique_users": count,
            "ue_lifecycles": lifecycles, "active_user_count_by_step": [count] * total_steps + [0],
            "connected_user_count_by_step": [count] * total_steps + [0],
            "population_config": {"profile": "constant"},
            "observation_config": {"type": "factored", "components": ["rsrp", "cio", "association", "active_mask"], "inactive_rsrp": -120.0},
            "reward_config": {"type": "load_balance"},
            "hysteresis_config": {"enabled": True, "value": 3.0},
            "osm_file_path": self.osm_file_path,
        }

    def _nearest_gate_node(self, source):
        """Cheapest reachable gate by road distance from `source`.

        Users head for the gate that is actually closest along the road network
        instead of a random one, which removes avoidable detours on the way out.
        """
        best_node, best_name, best_length = None, None, math.inf
        for name in sorted(self.gate_nodes_by_name):
            for node in self.gate_nodes_by_name[name]:
                if node == source:
                    return node, name, 0.0
                points = self._route(source, node)
                length = sum(distance(a, b) for a, b in zip(points, points[1:]))
                if length < best_length:
                    best_node, best_name, best_length = node, name, length
        if best_node is None:
            raise ValueError("No reachable exit gate from road node %s" % source)
        return best_node, best_name, best_length

    def _measure_walk_out_seconds(self, sample_nodes=200):
        """Mean road distance from a random interior point to its nearest gate.

        Reported in metres; the caller divides by the sampled speed range. It is
        a property of the shared map and gate set, so it is measured once per
        generator and reused: re-measuring per environment would fold node
        sampling noise into every environment's recorded population settings.
        """
        if sample_nodes not in self._walk_out_cache:
            nodes = list(self.interior_nodes)
            rng = np.random.default_rng(0)
            if sample_nodes and len(nodes) > sample_nodes:
                nodes = [int(node) for node in rng.choice(nodes, size=sample_nodes, replace=False)]
            lengths = []
            for node in nodes:
                _, _, best_length = self._nearest_gate_node(node)
                if np.isfinite(best_length):
                    lengths.append(best_length)
            if not lengths:
                raise ValueError("No interior node reaches a gate; check the gate configuration")
            self._walk_out_cache[sample_nodes] = (float(np.mean(lengths)), float(np.median(lengths)))
        return self._walk_out_cache[sample_nodes]

    def generate_population_scenario(
        self, max_active_users=100, total_steps=101, snapshot_seconds=10.0,
        simulation_seconds=1.0, control_seconds=None, seed=42, scenario_name=None,
        conditions=None, population_config=None, mobility_config=None,
        measurement_config=None, base_stations=None, connection_config=None,
        step_seconds=None,
    ):
        """Generate one map environment under a sampled set of conditions.

        `total_steps` counts snapshots (one control step each); `snapshot_seconds`
        is the control/observation interval and `simulation_seconds` is the
        physics interval. The scenario therefore simulates
        total_steps * snapshot_seconds of physical time while exposing only the
        snapshot sequence to the environment. `step_seconds` is accepted as an
        alias for `snapshot_seconds` by older callers. `control_seconds` may be a
        multiple of `snapshot_seconds` to coarsen control; it defaults to the
        snapshot interval.
        """
        self.load_osm_data()
        if step_seconds is not None:
            snapshot_seconds = step_seconds
        for name, value in (("snapshot_seconds", snapshot_seconds), ("simulation_seconds", simulation_seconds)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("%s must be finite and positive" % name)
        ratio = snapshot_seconds / simulation_seconds
        if abs(ratio - round(ratio)) > 1e-9 or round(ratio) < 1:
            raise ValueError("snapshot_seconds must be an integer multiple of simulation_seconds")
        snapshot_ticks = int(round(ratio))
        if control_seconds is None:
            control_seconds = snapshot_seconds
        control_seconds = float(control_seconds)
        control_ratio = control_seconds / snapshot_seconds
        if abs(control_ratio - round(control_ratio)) > 1e-9 or round(control_ratio) < 1:
            raise ValueError("control_seconds must be an integer multiple of snapshot_seconds")
        control_stride = int(round(control_ratio))
        if total_steps < 2:
            raise ValueError("At least two snapshots are required")
        total_time = total_steps * snapshot_seconds

        condition = dict(conditions or {})
        unknown = set(condition) - set(DEFAULT_CONDITION_CANDIDATES)
        if unknown:
            raise ValueError("Unknown environment conditions: %s" % sorted(unknown))
        # Defaults are conditions for a full-size environment; small fixtures
        # override them and are not expected to reach a stationary head-count.
        condition.setdefault("initial_users", min(45, max_active_users))
        condition.setdefault("arrival_rate_per_second", 0.02 if max_active_users >= 45 else 0.005)
        condition.setdefault("mean_exit_intent_seconds", 900.0 if max_active_users >= 45 else 120.0)
        condition.setdefault("speed_mps_range", [1.0, 1.8])
        condition.setdefault("west_entry_probability", 0.5)
        condition.setdefault("mean_connected_seconds", 600.0 if max_active_users >= 45 else 120.0)
        condition.update(population_config or {})
        initial_target = int(condition["initial_users"])
        arrival_rate = float(condition["arrival_rate_per_second"])
        mean_exit_intent = float(condition["mean_exit_intent_seconds"])
        mean_online = float(condition["mean_connected_seconds"])
        west_probability = float(condition["west_entry_probability"])
        if not 0 <= initial_target <= max_active_users:
            raise ValueError("initial_users must fit inside max_active_users")
        if not np.isfinite(arrival_rate) or arrival_rate < 0:
            raise ValueError("arrival_rate_per_second must be finite and nonnegative")
        if not np.isfinite(mean_exit_intent) or mean_exit_intent <= 0:
            raise ValueError("mean_exit_intent_seconds must be finite and positive")
        if not 0.0 <= west_probability <= 1.0:
            raise ValueError("west_entry_probability must lie in [0, 1]")

        mobility = {"speed_mps_range": list(condition["speed_mps_range"])}
        if set(mobility_config or {}) - set(mobility):
            raise ValueError("Unknown mobility configuration")
        # An explicit mobility_config overrides the sampled condition, which is
        # what small fixtures use to pin the walking speed.
        mobility.update(mobility_config or {})
        for name, bounds in mobility.items():
            if len(bounds) != 2 or not np.all(np.isfinite(bounds)) or bounds[0] <= 0 or bounds[1] < bounds[0]:
                raise ValueError("Invalid range: %s" % name)

        connection = {"mean_connected_seconds": mean_online, "disconnected_seconds_range": [30.0, 90.0]}
        if set(connection_config or {}) - set(connection):
            raise ValueError("Unknown connection configuration")
        connection.update(connection_config or {})
        offline = connection["disconnected_seconds_range"]
        if not np.isfinite(connection["mean_connected_seconds"]) or connection["mean_connected_seconds"] < 0:
            raise ValueError("mean_connected_seconds must be finite and nonnegative")
        if len(offline) != 2 or not np.all(np.isfinite(offline)) or offline[0] <= 0 or offline[1] < offline[0]:
            raise ValueError("Invalid disconnected_seconds_range")

        stations = self._convert_base_stations_to_meters(base_stations or DEFAULT_BASE_STATIONS)
        if len({s["id"] for s in stations}) != len(stations):
            raise ValueError("Base station IDs must be unique")
        measurement = {"version": 1, "mode": "full", "seed": seed,
                       "report_count_probs": [1 / len(stations)] * len(stations),
                       "probability_source": "synthetic_default_uniform"}
        measurement.update(measurement_config or {})
        if "report_count_probs" in (measurement_config or {}) and "probability_source" not in measurement_config:
            measurement["probability_source"] = "explicit_configuration"
        # Validate at generation time as well as at environment construction.
        from env.measurement import MeasurementReporter
        MeasurementReporter(measurement, [s["id"] for s in stations])

        rng_arrival = child_rng(seed, "arrivals")
        gate_names = sorted(self.gate_nodes_by_name)

        def pick_entry_node(rng):
            if len(gate_names) > 1:
                if "West_Gate" in self.gate_nodes_by_name and "North_Gate" in self.gate_nodes_by_name:
                    name = "West_Gate" if rng.random() < west_probability else "North_Gate"
                else:
                    name = str(rng.choice(gate_names))
            else:
                name = gate_names[0]
            return name, int(rng.choice(self.gate_nodes_by_name[name]))

        active = {}
        records = []
        arrivals_waiting = 0
        stat_names = ("attempted_arrivals", "entered", "left", "exit_intents",
                      "waiting_arrivals", "departing", "present")
        stats = {name: [] for name in stat_names}

        def spawn(second, initial=False):
            uid = len(records)
            rng_route = child_rng(seed, "route", uid)
            rng_speed = child_rng(seed, "speed", uid)
            rng_intent = child_rng(seed, "exit_intent", uid)
            interior_node = int(rng_route.choice(self.interior_nodes))
            entry_name, entry_node = pick_entry_node(rng_route)
            start = interior_node if initial else entry_node
            spawn_step = 0 if initial else min(total_steps, math.ceil(second / snapshot_seconds - 1e-9))
            # Positions are appended by the snapshot loop, which records every
            # user present at each snapshot index exactly once. The census covers
            # indices 0..total_steps-1, so a user that never leaves is present
            # through the last one and its exclusive bound is total_steps.
            record = {
                "ue_id": uid, "spawn_step": spawn_step, "leave_step": total_steps,
                "spawn_second": float(second), "initial_population": initial,
                "entry_gate": None if initial else entry_name,
                "entry_node": None if initial else entry_node, "initial_node": start,
                "exit_gate": None, "exit_node": None,
                "exit_intent_second": float(second + rng_intent.exponential(mean_exit_intent)),
                "end_reason": "horizon", "trajectory": [], "disconnect_intervals": [],
            }
            user = {
                "record": record, "position": self.node_positions_cache[start],
                "node": start, "previous_node": None, "exit_node": None,
                "state": "walking", "walk_rng": child_rng(seed, "walk", uid),
                "speed": float(rng_speed.uniform(*mobility["speed_mps_range"])),
            }
            self._choose_walk_edge(user)
            if initial and rng_route.random() < 0.25:
                # Seed part of the initial cohort already walking out, so the
                # opening snapshot is not always arrivals-only. They advance by
                # less than one snapshot of walking, which keeps them on the map
                # in snapshot 0 instead of leaving before the episode starts.
                self._start_departure(user)
                if user["route"]:
                    total_length = sum(distance(a, b) for a, b in zip(user["route"], user["route"][1:]))
                    progress = min(total_length, 0.7 * float(user["speed"]) * snapshot_seconds)
                    self._advance(user, progress * float(rng_route.uniform(0.0, 1.0)))
            record["initial_phase"] = user["state"]
            record["speed_mps"] = user["speed"]
            records.append(record)
            active[uid] = user

        for _ in range(initial_target):
            spawn(0.0, initial=True)
        # Snapshot 0 is the initial state: position, head-count and counters all
        # describe the same instant that the environment reads at step 0.
        for user in active.values():
            user["record"]["trajectory"].append({"step": 0, "position": list(user["position"])})
        stats["present"].append(len(active))
        interval_stats = {name: [] for name in stat_names if name != "present"}
        # Counter slot 0 belongs to snapshot 0, which has no preceding interval.
        for name in interval_stats:
            stats[name] = [0]

        # The episode holds `total_steps` snapshots (indices 0..total_steps-1) and
        # therefore `total_steps - 1` control transitions, matching the
        # environment, which stops at the last snapshot.
        for index in range(total_steps - 1):
            left = 0
            intents = 0
            entered = 0
            attempted = int(rng_arrival.poisson(arrival_rate * snapshot_seconds))
            arrivals_waiting += attempted
            for tick in range(snapshot_ticks):
                second = index * snapshot_seconds + tick * simulation_seconds
                for uid, user in list(active.items()):
                    record = user["record"]
                    if user["state"] == "walking":
                        self._advance_walk(user, user["speed"] * simulation_seconds)
                    else:
                        self._advance(user, user["speed"] * simulation_seconds)
                    if user["state"] == "walking" and second >= record["exit_intent_second"]:
                        self._start_departure(user)
                        intents += 1
                    # A user whose exit route is exhausted has reached the gate.
                    # This is checked after the intent switch as well, because the
                    # intent can fire on the same tick that finishes the route.
                    if user["state"] == "departing" and user["route_index"] >= len(user["route"]):
                        # exit_second is the instant the user reaches the gate. The
                        # user is present at the snapshot index that interval starts
                        # from, and absent from every later one, so leave_step is
                        # one past that index. Positions are written at the end of
                        # each interval, making [spawn_step, leave_step) exactly the
                        # recorded range that the census also counts.
                        exit_second = second + simulation_seconds
                        # The user is absent from the first snapshot after the exit,
                        # which is ceil(exit_second / snapshot) for an exit that
                        # falls between two snapshots. Snapshots are indexed
                        # 0..total_steps-1, so the bound saturates at total_steps.
                        first_absent = int(math.ceil(exit_second / snapshot_seconds - 1e-9))
                        record["leave_step"] = min(total_steps, max(1, first_absent))
                        record["end_reason"] = "exit"
                        record["exit_second"] = float(exit_second)
                        record["exit_position"] = list(user["position"])
                        record["exit_gate"] = user.get("exit_gate")
                        record["exit_node"] = user.get("exit_node")
                        del active[uid]
                        left += 1
                # Capacity is a hard cap on present users: arrivals queue outside
                # the map and disconnection never frees a place.
                admitted = min(arrivals_waiting, max_active_users - len(active))
                for _ in range(admitted):
                    spawn(second + simulation_seconds)
                arrivals_waiting -= admitted
                entered += admitted

            values = (attempted, entered, left, intents, arrivals_waiting,
                      sum(1 for user in active.values() if user["state"] == "departing"))
            for name, value in zip(interval_stats, values):
                interval_stats[name].append(int(value))

            # One joint action at the end of the interval: census the users that
            # are present at snapshot `index + 1` and record their positions
            # there. Users that left during the interval are already gone, so
            # their trajectory stops at the previous snapshot, which is exactly
            # the [spawn_step, leave_step) range they are queried over.
            stats["present"].append(len(active))
            for name, values in interval_stats.items():
                stats[name] = [0] + values
            # Everyone still present is at snapshot `index + 1` once the interval
            # has been simulated: this single recording makes each user's
            # trajectory cover exactly [spawn_step, leave_step).
            for user in active.values():
                user["record"]["trajectory"].append(
                    {"step": index + 1, "position": list(user["position"])})

        _, _, width, height = self._get_scene_bounds_meters()
        visible_counts = np.array(stats["present"], dtype=int)
        for record in records:
            record["disconnect_intervals"] = self._disconnect_intervals(
                seed, record, simulation_seconds, snapshot_seconds, connection)
            for interval in record["disconnect_intervals"]:
                visible_counts[interval["start_step"]:interval["end_step"]] -= 1
        visible_counts = np.clip(visible_counts, 0, None)
        stats["online"] = visible_counts.tolist()
        stats["disconnected"] = (np.array(stats["present"]) - visible_counts).tolist()

        walk_distance_mean, walk_distance_median = self._measure_walk_out_seconds()
        walk_out_seconds = {
            "mean": walk_distance_mean / float(np.mean(mobility["speed_mps_range"])),
            "median": walk_distance_median / float(np.mean(mobility["speed_mps_range"])),
            "source": "measured_on_loaded_road_network",
        }
        scenario_name = scenario_name or "map_env_%d" % seed
        population_config_out = {
            key: value for key, value in condition.items()
            if key not in ("condition_registry", "sampled_at_rng_state")
        }
        population_config_out.update({
            "initial_users_actual": initial_target,
            "arrival_rate_per_second": arrival_rate,
            "mean_exit_intent_seconds": mean_exit_intent,
            "mean_walk_out_seconds": walk_out_seconds["mean"],
            "mean_presence_seconds": mean_exit_intent + walk_out_seconds["mean"],
            "turnover_per_environment": total_time / (mean_exit_intent + walk_out_seconds["mean"]),
        })
        config = {
            "schema_version": "ue_lifecycle_v1", "scenario_type": "map",
            "scenario_id": scenario_name, "generator_version": "map_road_env_v3",
            "seed": int(seed), "measurement_scenario_key": "map_seed_%d" % seed,
            "osm_file_path": self.osm_file_path,
            "scene": {"width": width, "height": height, "total_steps": total_steps,
                      "step_seconds": snapshot_seconds, "snapshot_seconds": snapshot_seconds,
                      "simulation_seconds": simulation_seconds, "control_seconds": control_seconds,
                      "total_time": total_time},
            "base_stations": stations, "max_active_users": int(max_active_users),
            "total_unique_users": len(records), "ue_lifecycles": records,
            "population_config": population_config_out,
            "mobility_config": mobility, "connection_config": connection, "measurement_config": measurement,
            "gate_config": self.gates_config, "gate_nodes": self.gate_nodes_by_name,
            "population_statistics": stats,
            "walk_out_seconds": walk_out_seconds,
            "present_user_count_by_step": stats["present"],
            "active_user_count_by_step": stats["online"],
            "connected_user_count_by_step": stats["online"],
            "censored_user_count": len(active),
            "observation_config": {"type": "factored", "inactive_rsrp": -120.0,
                                   "components": ["rsrp", "measurement_mask", "cio", "association", "active_mask"]},
            "reward_config": {"type": "load_balance"},
            "hysteresis_config": {"enabled": True, "value": 2.0},
            "cio_config": {"min_value": -6, "max_value": 6, "step_size": 0.5},
        }
        return config

    @staticmethod
    def save_scenario_to_file(config, output_path):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def generate_map_scenario(osm_file_path=MAP_PATH, output_path=None, gates=None, **kwargs):
    generator = OSMScenarioGenerator(osm_file_path, gates=gates)
    config = generator.generate_population_scenario(**kwargs)
    if output_path is not None:
        generator.save_scenario_to_file(config, output_path)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Generate road-constrained map environments on sdnu.osm")
    parser.add_argument("--osm-file", default=str(MAP_PATH))
    parser.add_argument("--max-active-users", type=int, default=100)
    parser.add_argument("--total-steps", type=int, default=101,
                        help="Snapshots kept at 0,10,...,1000 s; 101 snapshots give 100 control steps")
    parser.add_argument("--snapshot-seconds", type=float, default=10.0,
                        help="Observation and control interval")
    parser.add_argument("--simulation-seconds", type=float, default=1.0,
                        help="Physics interval for movement, entering and connectivity")
    parser.add_argument("--control-seconds", type=float, default=None,
                        help="Control interval; default equals --snapshot-seconds. "
                             "May be a multiple of the snapshot interval to coarsen control.")
    parser.add_argument("--num-scenarios", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--output-dir", default="map_based_scenarios/scenarios")
    parser.add_argument("--config",
                        help="Optional YAML overriding condition_candidates, conditions, "
                             "mobility_config, connection_config, measurement_config, gates")
    parser.add_argument("--measurement-mode", choices=["full", "sparse"], default=None)
    parser.add_argument("--report-count-probs", nargs="+", type=float)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_scenarios < 1 or args.seed < 0:
        parser.error("num-scenarios must be positive and seed nonnegative")

    settings = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    settings = settings or {}
    known = {"gates", "condition_candidates", "conditions", "population_config",
             "mobility_config", "connection_config", "measurement_config"}
    unknown = set(settings) - known
    if unknown:
        parser.error("Unknown configuration keys: %s" % sorted(unknown))

    generator = OSMScenarioGenerator(args.osm_file, gates=settings.get("gates"))
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    conditions = sample_environment_conditions(
        args.num_scenarios, args.seed,
        candidates=settings.get("condition_candidates"),
        overrides=settings.get("conditions") or settings.get("population_config"),
    )
    entries = []
    for condition in conditions:
        index, seed = condition["index"], condition["seed"]
        name = "%s_map_env_%d_%03d" % (args.split, args.seed, index)
        output = root / (name + ".yaml")
        if output.exists() and not args.overwrite:
            raise FileExistsError(output)
        measurement = dict(settings.get("measurement_config", {}))
        if args.measurement_mode:
            measurement["mode"] = args.measurement_mode
        if args.report_count_probs is not None:
            measurement["report_count_probs"] = args.report_count_probs
            measurement["probability_source"] = "explicit_configuration"
        config = generator.generate_population_scenario(
            max_active_users=args.max_active_users, total_steps=args.total_steps,
            snapshot_seconds=args.snapshot_seconds, simulation_seconds=args.simulation_seconds,
            control_seconds=args.control_seconds,
            seed=seed, scenario_name=name,
            conditions={key: condition[key] for key in CONDITION_ORDER},
            mobility_config=settings.get("mobility_config"), measurement_config=measurement,
            connection_config=settings.get("connection_config"),
        )
        config["scenario_split"] = args.split
        config["conditions_seed"] = condition["conditions_seed"]
        config["condition_registry"] = condition["condition_registry"]
        generator.save_scenario_to_file(config, output)

        stats = config["population_statistics"]
        present = config["present_user_count_by_step"]
        online = config["active_user_count_by_step"]
        record = {
            "scenario_id": name, "path": str(output), "seed": seed,
            "conditions": {key: config["population_config"][key]
                           for key in CONDITION_ORDER},
            "present_min": min(present), "present_max": max(present),
            "present_mean": float(np.mean(present)),
            "online_mean": float(np.mean(online)),
            "final_present": present[-1],
            "entered_total": int(sum(stats["entered"])),
            "left_total": int(sum(stats["left"])),
            "censored_users": config["censored_user_count"],
            "disconnect_intervals": int(sum(len(r["disconnect_intervals"]) for r in config["ue_lifecycles"])),
            "at_capacity_steps": int(sum(1 for value in present if value >= args.max_active_users)),
            "mean_speed_mps": float(np.mean([r["speed_mps"] for r in config["ue_lifecycles"]])),
            "measurement_config": config["measurement_config"],
        }
        entries.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    manifest = root / ("%s_seed%d_manifest.json" % (args.split, args.seed))
    effective_control = args.control_seconds or args.snapshot_seconds
    manifest.write_text(json.dumps(
        {"seed": args.seed, "split": args.split, "num_environments": len(entries),
         "generator_version": "map_road_env_v3",
         "scene": {"total_steps": args.total_steps, "snapshot_seconds": args.snapshot_seconds,
                   "simulation_seconds": args.simulation_seconds,
                   "control_seconds": effective_control,
                   "total_time": args.total_steps * args.snapshot_seconds,
                   "control_steps": int((args.total_steps - 1) // round(effective_control / args.snapshot_seconds))},
         "condition_candidates": settings.get("condition_candidates") or DEFAULT_CONDITION_CANDIDATES,
         "environments": entries}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved manifest: %s" % manifest, flush=True)


if __name__ == "__main__":
    # Allow direct script execution as well as importing from collection tools.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
