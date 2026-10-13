import copy
import math
import os
import warnings
from collections import deque

import gym
import matplotlib.pyplot as plt
import numpy as np

from .factored_mlb import FactoredMLBMixin
from .scenario_loader import Scenario, load_scenario_config


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"

try:
    import osmnx as ox

    OSM_AVAILABLE = True
except ImportError as error:
    warnings.warn(
        "Warning: Unable to import OSM-related libraries, "
        f"falling back to simplified mode: {error}",
        RuntimeWarning,
    )
    OSM_AVAILABLE = False


class CIOLoadBalancingEnv(FactoredMLBMixin, gym.Env):
    """CIO load-balancing environment for UE lifecycle scenarios.

    Active UEs keep stable slots for the duration of their lifecycle. Empty
    slots are padded so that observations remain fixed-size when users spawn,
    leave, or temporarily disconnect. An action selected from population
    snapshot ``P_t`` updates the CIO state and is evaluated on the following
    population snapshot ``P_{t+1}``.
    """

    metadata = {"render_modes": ["human"]}
    DEFAULT_OBSERVATION_COMPONENTS = (
        "rsrp",
        "cio",
        "association",
        "active_mask",
    )

    def __init__(self, config, scenario=None):
        super().__init__()
        self.config = load_scenario_config(config)
        self.scenario = scenario if scenario is not None else Scenario(self.config)

        self.scene_width = float(self.config["scene"]["width"])
        self.scene_height = float(self.config["scene"]["height"])
        self.total_time = int(self.scenario.total_steps)
        # Number of snapshots advanced per step(): 1 when the control interval
        # equals the snapshot interval, larger when control is coarser.
        self.control_stride = int(getattr(self.scenario, "control_stride", 1))
        self.control_steps = int(getattr(self.scenario, "control_steps", self.total_time - 1))
        self.base_stations = copy.deepcopy(self.scenario.base_stations)
        self.bs_by_id = {int(bs["id"]): bs for bs in self.base_stations}
        from .measurement import MeasurementReporter

        self.measurement_reporter = MeasurementReporter(
            self.config.get("measurement_config"), self.bs_by_id,
            self.config.get("measurement_scenario_key", self.config.get("scenario_id", "")),
        )
        self.last_measurement_reassociation_count = 0
        self.max_active_users = int(self.scenario.max_active_users)
        if self.max_active_users <= 0:
            raise ValueError("Scenarios must define at least one active UE slot")

        self.active_devices = []
        self.device_slots = [None] * self.max_active_users
        self.slot_by_ue_id = {}
        self.connected_bs_by_ue_id = {}
        self.previous_connected_bs_by_ue_id = {}

        self._configure_factored_spaces()
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32,
        )

        network_config = self.config.get("network_config", {})
        self.blocked_bps = float(network_config.get("blocked_bps", 512 * 1000))
        self.min_rsrp_mcs = float(network_config.get("min_rsrp_mcs", -110))
        self.max_rsrp_mcs = float(network_config.get("max_rsrp_mcs", -65))
        self.f_c = float(network_config.get("carrier_frequency_mhz", 2000))
        self.h_b = float(network_config.get("bs_antenna_height_m", 30))
        self.h_m = float(network_config.get("ue_antenna_height_m", 1.5))
        self.default_tx_power_dbm = float(network_config.get("tx_power_dbm", 20))
        self.g_tx = float(network_config.get("bs_antenna_gain_dbi", 14))
        self.g_rx = float(network_config.get("ue_antenna_gain_dbi", 0))
        self.cable_loss = float(network_config.get("cable_loss_db", 2))

        self.cio_matrix = {}
        self.current_time = 0
        self.last_handover_count = 0
        self.last_pingpong_count = 0
        self.last_handover_baseline = 0
        self.last_pingpong_baseline = 0
        self.last_cio_change_magnitude = 0.0
        self.record_association_diagnostics = False
        self.last_association_diagnostics = None

        self.fig = None
        self.ax = None
        self.population_ax = None
        self._sidebar_texts = []
        self.rendering = False
        self.save_animation_path = self.config.get("save_animation_path")
        self.frames = []

        self.osm_file_path = self.config.get("osm_file_path")
        self.osm_graph = None
        self.road_nodes = None
        self.road_edges = None
        self.map_loaded = False

        self.throughput_history = deque(maxlen=100)
        self.fairness_history = deque(maxlen=100)
        self.blocking_history = deque(maxlen=100)
        self.average_load_history = deque(maxlen=100)
        self.population_history = []
        self.present_population_history = []
        self.device_trajectories = {
            int(lifecycle["ue_id"]): deque(maxlen=50)
            for lifecycle in self.scenario.lifecycles
        }

        self._reset_cio_matrix()
        self._refresh_active_devices()
        self._measure_rsrp()
        self._calculate_load()
        self._record_visualization_metrics()
        self._record_active_positions()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_time = 0
        self.last_association_diagnostics = None
        self.connected_bs_by_ue_id = {}
        self.previous_connected_bs_by_ue_id = {}
        self.slot_by_ue_id = {}
        self.device_slots = [None] * self.max_active_users
        self.throughput_history.clear()
        self.fairness_history.clear()
        self.blocking_history.clear()
        self.average_load_history.clear()
        self.population_history.clear()
        self.present_population_history.clear()
        self.frames.clear()
        for trajectory in self.device_trajectories.values():
            trajectory.clear()
        self._reset_cio_matrix()
        self._reset_bs_state()
        self._refresh_active_devices()
        self._measure_rsrp()
        self._calculate_load()
        self._reset_transition_metrics()
        self._record_visualization_metrics()
        self._record_active_positions()
        return self._get_observation(), self._build_info()

    def step(self, action):
        if self.current_time >= self.total_time - 1:
            raise RuntimeError(
                "Episode is finished because no next population snapshot is "
                "available; call reset() before step()"
            )

        old_cio_matrix = self.cio_matrix.copy()
        self._apply_cio_action(action)

        # The controller observes P_t, but the updated CIO takes effect during
        # the following control interval. Advance to the next control point
        # before computing associations, load, and reward. The stride is 1 when
        # the control interval equals the snapshot interval; when the stride does
        # not divide the horizon evenly the final interval is shortened so that
        # the episode still ends exactly on the last snapshot.
        self.current_time = min(self.current_time + self.control_stride, self.total_time - 1)
        self._refresh_active_devices()
        self._measure_rsrp()
        self._calculate_load()
        pre_action_loads = self._get_bs_loads()

        self.last_handover_baseline, self.last_pingpong_baseline = (
            self._get_hypothetical_connections(old_cio_matrix)
        )
        self.last_association_diagnostics = None
        if self.record_association_diagnostics:
            diagnostics, predicted_connections = self._association_change_diagnostics(
                old_cio_matrix
            )
        self._update_connections()
        if self.record_association_diagnostics:
            actual_connections = {
                int(device["id"]): (
                    int(device["connected_bs"]["id"])
                    if device.get("connected_bs") is not None else None
                )
                for device in self.active_devices
                if device.get("connected_allowed", True)
            }
            if actual_connections != predicted_connections:
                raise RuntimeError("Association diagnostic differs from the actual update")
            self.last_association_diagnostics = diagnostics
        self._calculate_load()

        throughput = self._calculate_throughput()
        fairness = self._calculate_fairness()
        blocking = self._calculate_blocking()
        reward = self._calculate_reward(throughput, fairness, blocking)
        info = self._build_info(
            throughput=throughput,
            fairness=fairness,
            blocking=blocking,
            reward=reward,
            pre_action_loads=pre_action_loads,
        )
        if self.last_association_diagnostics is not None:
            info["association_diagnostics"] = dict(self.last_association_diagnostics)
        self._record_visualization_metrics(throughput, fairness, blocking)

        done = self.current_time >= self.total_time - 1
        self._record_active_positions()

        return self._get_observation(), reward, done, False, info

    def get_current_info(self):
        self._calculate_load()
        return self._build_info()

    def _reset_cio_matrix(self):
        self.cio_matrix = {
            (int(bs_i["id"]), int(bs_j["id"])): 0.0
            for bs_i in self.base_stations
            for bs_j in self.base_stations
            if int(bs_i["id"]) != int(bs_j["id"])
        }

    def _reset_bs_state(self):
        for bs in self.base_stations:
            bs["connected_devices"] = []
            bs["load"] = 0.0
            bs["throughput"] = 0.0
            bs["resource_utilization"] = 0.0
            bs["offered_load"] = 0.0
            bs["served_utilization"] = 0.0
            bs["required_prbs"] = 0
            bs["mcs_distribution"] = [0] * 10

    def _refresh_active_devices(self):
        users = self.scenario.active_users_at(self.current_time)
        users_by_id = {int(user["ue_id"]): user for user in users}
        active_ue_ids = set(users_by_id)

        for ue_id in list(self.slot_by_ue_id):
            if ue_id not in active_ue_ids:
                slot = self.slot_by_ue_id.pop(ue_id)
                self.device_slots[slot] = None
                self.connected_bs_by_ue_id.pop(ue_id, None)
                self.previous_connected_bs_by_ue_id.pop(ue_id, None)

        free_slots = [
            slot for slot, device in enumerate(self.device_slots) if device is None
        ]
        for ue_id in sorted(active_ue_ids):
            if ue_id in self.slot_by_ue_id:
                continue
            if not free_slots:
                raise RuntimeError(
                    "Active UE count exceeds max_active_users declared by the scenario"
                )
            self.slot_by_ue_id[ue_id] = free_slots.pop(0)

        self.active_devices = []
        for ue_id, user in users_by_id.items():
            initial_bs_id = user.get("initial_base_station_id")
            connected_bs_id = self.connected_bs_by_ue_id.get(ue_id, initial_bs_id)
            connected_bs = (
                self.bs_by_id.get(int(connected_bs_id))
                if connected_bs_id is not None
                else None
            )
            connected_allowed = bool(user["connected_allowed"])
            if not connected_allowed:
                connected_bs = None

            device = {
                "id": ue_id,
                "slot": self.slot_by_ue_id[ue_id],
                "current_position": list(user["position"]),
                "connected_allowed": connected_allowed,
                "connected_bs": connected_bs,
                "previous_connected_bs_id": self.previous_connected_bs_by_ue_id.get(
                    ue_id
                ),
                "recent_handover": 0.0,
                "rsrp_measurements": {
                    int(bs_id): float(rsrp)
                    for bs_id, rsrp in user.get("rsrp_measurements", {}).items()
                },
                "measurement_mask": {
                    int(bs_id): bool(measured)
                    for bs_id, measured in user.get("measurement_mask", {}).items()
                },
                "required_prbs": 0,
                "rate_per_prb": 0.0,
                "achieved_rate": 0.0,
            }
            self.device_slots[device["slot"]] = device
            self.active_devices.append(device)

        self.active_devices.sort(key=lambda device: device["slot"])
        self._assign_connected_devices()

    def _assign_connected_devices(self):
        for bs in self.base_stations:
            bs["connected_devices"] = []
        for device in self.active_devices:
            connected_bs = device.get("connected_bs")
            if connected_bs is None or not device.get("connected_allowed", True):
                continue
            connected_bs["connected_devices"].append(device)
            self.connected_bs_by_ue_id[device["id"]] = int(connected_bs["id"])

    def _get_observation(self):
        return self._build_observation()

    def _calculate_path_loss_cost_hata(self, distance_m):
        d_km = max(distance_m / 1000.0, 0.02)
        antenna_correction = (1.1 * math.log10(self.f_c) - 0.7) * self.h_m - (
            1.56 * math.log10(self.f_c) - 0.8
        )
        return (
            46.3
            + 33.9 * math.log10(self.f_c)
            - 13.82 * math.log10(self.h_b)
            - antenna_correction
            + (44.9 - 6.55 * math.log10(self.h_b)) * math.log10(d_km)
        )

    def _measure_rsrp(self):
        self.last_measurement_reassociation_count = 0
        for device in self.active_devices:
            device["rsrp_measurements"] = {}
            for bs in self.base_stations:
                distance = math.sqrt(
                    (device["current_position"][0] - bs["position"][0]) ** 2
                    + (device["current_position"][1] - bs["position"][1]) ** 2
                )
                tx_power = float(bs.get("tx_power", self.default_tx_power_dbm))
                path_loss = self._calculate_path_loss_cost_hata(distance)
                device["rsrp_measurements"][int(bs["id"])] = float(
                    tx_power + self.g_tx + self.g_rx - path_loss - self.cable_loss
                )
        self._apply_measurement_reports()

    def _apply_measurement_reports(self):
        """Restrict all control paths to reported values before computing load."""
        for device in self.active_devices:
            full = dict(device["rsrp_measurements"])
            device["full_rsrp_measurements"] = full
            measured, mask = self.measurement_reporter.report(full, device["id"])
            device["rsrp_measurements"] = measured
            device["measurement_mask"] = mask
            current = device.get("connected_bs")
            initialize = current is None and self.config.get("scenario_type") == "map"
            invalid = current is not None and int(current["id"]) not in measured
            if device.get("connected_allowed", True) and (initialize or invalid):
                best = min(measured, key=lambda cell: (-measured[cell], cell))
                device["connected_bs"] = self.bs_by_id[best]
                self.connected_bs_by_ue_id[device["id"]] = best
                if invalid:
                    self.last_measurement_reassociation_count += 1
                    # Measurement repair is separate from CIO handover/ping-pong.
                    self.previous_connected_bs_by_ue_id.pop(device["id"], None)
                    device["previous_connected_bs_id"] = None
        self._assign_connected_devices()

    def _association_change_diagnostics(self, old_cio_matrix):
        """Compare configurations on one snapshot without changing user state.

        Every prediction starts from the same references available immediately
        before the ordinary connection update. References are snapshot-local in
        MRO and may carry simulator connection state in map scenarios.
        """
        current = self._get_hypothetical_connections(
            self.cio_matrix, return_connections=True
        )
        previous = self._get_hypothetical_connections(
            old_cio_matrix, return_connections=True
        )
        zero = self._get_hypothetical_connections(
            {link: 0.0 for link in self.cio_matrix}, return_connections=True
        )
        references = {
            int(device["id"]): (
                int(device["connected_bs"]["id"])
                if device.get("connected_bs") is not None else None
            )
            for device in self.active_devices
            if device.get("connected_allowed", True)
        }
        eligible = [ue_id for ue_id, cell in current.items() if cell is not None]
        if any(previous[ue_id] is None or zero[ue_id] is None for ue_id in eligible):
            raise RuntimeError("Counterfactual branches have different observable users")
        diagnostics = {"eligible_users": len(eligible)}
        for name, baseline in (
            ("previous_cio", previous), ("zero_cio", zero), ("reference", references)
        ):
            diagnostics["changed_vs_" + name] = sum(
                current[ue_id] != baseline[ue_id] for ue_id in eligible
            )
        diagnostics["previous_vs_zero_cio"] = sum(
            previous[ue_id] != zero[ue_id] for ue_id in eligible
        )
        diagnostics["zero_cio_vs_reference"] = sum(
            zero[ue_id] != references[ue_id] for ue_id in eligible
        )
        return diagnostics, current

    def _update_connections(self):
        self.last_handover_count = 0
        self.last_pingpong_count = 0
        hysteresis_config = self.config.get("hysteresis_config", {})
        hysteresis_enabled = bool(hysteresis_config.get("enabled", True))
        hysteresis_value = float(hysteresis_config.get("value", 2.0))

        for device in self.active_devices:
            device["recent_handover"] = 0.0
            if not device.get("connected_allowed", True):
                device["connected_bs"] = None
                continue

            current_bs = device.get("connected_bs")
            best_bs = None
            best_score = -np.inf
            for bs in self.base_stations:
                bs_id = int(bs["id"])
                if bs_id not in device["rsrp_measurements"]:
                    continue
                rsrp = device["rsrp_measurements"][bs_id]
                if current_bs is not None and int(current_bs["id"]) != bs_id:
                    current_bs_id = int(current_bs["id"])
                    cio = self.cio_matrix[(current_bs_id, bs_id)] - self.cio_matrix[
                        (bs_id, current_bs_id)
                    ]
                else:
                    cio = 0.0

                score = rsrp + cio
                if (
                    hysteresis_enabled
                    and current_bs is not None
                    and bs_id == int(current_bs["id"])
                ):
                    score += hysteresis_value
                if score > best_score:
                    best_score = score
                    best_bs = bs

            current_bs_id = int(current_bs["id"]) if current_bs is not None else None
            best_bs_id = int(best_bs["id"]) if best_bs is not None else None
            if current_bs_id != best_bs_id:
                self.last_handover_count += 1
                device["recent_handover"] = 1.0
                if device.get("previous_connected_bs_id") == best_bs_id:
                    self.last_pingpong_count += 1
                self.previous_connected_bs_by_ue_id[device["id"]] = current_bs_id
                device["previous_connected_bs_id"] = current_bs_id

            device["connected_bs"] = best_bs
            if best_bs_id is not None:
                self.connected_bs_by_ue_id[device["id"]] = best_bs_id

        self._assign_connected_devices()

    def _mcs_index_from_rsrp(self, rsrp):
        if rsrp >= self.max_rsrp_mcs:
            return 9
        if rsrp <= self.min_rsrp_mcs:
            return 0
        return int(
            9
            * (rsrp - self.min_rsrp_mcs)
            / (self.max_rsrp_mcs - self.min_rsrp_mcs)
        )

    def _calculate_load(self):
        total_prbs = self._total_prbs()
        user_demand_bps = self._user_demand_bps()
        prb_bandwidth = self._prb_bandwidth()

        for device in self.active_devices:
            device["achieved_rate"] = 0.0

        for bs in self.base_stations:
            bs["throughput"] = 0.0
            bs["mcs_distribution"] = [0] * 10
            required_prbs = 0

            for device in bs["connected_devices"]:
                rsrp = device["rsrp_measurements"].get(int(bs["id"]), -120.0)
                mcs_index = self._mcs_index_from_rsrp(rsrp)
                bs["mcs_distribution"][mcs_index] += 1
                rate_per_prb = (0.5 + mcs_index * 0.5) * prb_bandwidth
                device_required_prbs = (
                    math.ceil(user_demand_bps / rate_per_prb)
                    if rate_per_prb > 0
                    else total_prbs
                )
                device["required_prbs"] = device_required_prbs
                device["rate_per_prb"] = rate_per_prb
                required_prbs += device_required_prbs

            offered_load = required_prbs / total_prbs
            bs["required_prbs"] = required_prbs
            bs["offered_load"] = offered_load
            bs["served_utilization"] = min(offered_load, 1.0)
            bs["resource_utilization"] = offered_load
            bs["load"] = offered_load

            scaling_factor = min(1.0, total_prbs / required_prbs) if required_prbs else 1.0
            for device in bs["connected_devices"]:
                device["achieved_rate"] = min(
                    user_demand_bps,
                    device["required_prbs"]
                    * scaling_factor
                    * device["rate_per_prb"],
                )
                bs["throughput"] += device["achieved_rate"]

    def _get_bs_loads(self):
        return [float(bs["resource_utilization"]) for bs in self.base_stations]

    def _calculate_throughput(self):
        return float(sum(bs["throughput"] for bs in self.base_stations))

    def _calculate_fairness(self):
        throughputs = [
            float(bs["throughput"])
            for bs in self.base_stations
            if bs["throughput"] > 0
        ]
        if not throughputs:
            return 1.0
        return float(
            sum(throughputs) ** 2
            / (len(throughputs) * sum(value**2 for value in throughputs))
        )

    def _calculate_blocking(self):
        if not self.active_devices:
            return 0.0
        blocked = sum(
            1
            for device in self.active_devices
            if device.get("connected_bs") is None
            or device.get("achieved_rate", 0.0) < self.blocked_bps
        )
        return float(blocked / len(self.active_devices))

    def _calculate_reward(self, throughput, fairness, blocking):
        reward_config = self.config.get("reward_config", {})
        reward_type = reward_config.get("type", "structured_mlb")

        if reward_type == "throughput":
            max_possible = self.num_base_stations * self._max_throughput_per_bs()
            return float(throughput / max_possible) if max_possible > 0 else 0.0
        if reward_type == "non_blocked":
            return float(1.0 - blocking)
        if reward_type == "load_balance":
            loads = self._get_bs_loads()
            return float(-np.std(loads)) if loads else 0.0
        if reward_type in ("structured_mlb", "mlb_structured"):
            return float(self._structured_mlb_reward(throughput, blocking))
        if reward_type == "structured_mlb_cf_debiased":
            return float(
                self._structured_mlb_reward(
                    throughput, blocking, use_cf_debias=True
                )
            )
        if reward_type == "throughput_and_balance":
            max_possible = self.num_base_stations * self._max_throughput_per_bs()
            normalized_throughput = throughput / max_possible if max_possible > 0 else 0.0
            loads = self._get_bs_loads()
            return float(normalized_throughput + 1.0 - np.std(loads))

        alpha = float(reward_config.get("alpha", 0.5))
        beta = float(reward_config.get("beta", 0.3))
        gamma = float(reward_config.get("gamma", 0.2))
        return float(alpha * throughput + beta * fairness - gamma * blocking)

    def _build_info(
        self,
        throughput=None,
        fairness=None,
        blocking=None,
        reward=None,
        pre_action_loads=None,
    ):
        loads = self._get_bs_loads()
        active_mask = self._get_active_mask_observation()
        connected_mask = self._get_connected_mask_observation()
        slot_ue_ids = [
            int(device["id"]) if device is not None else -1
            for device in self.device_slots
        ]
        return {
            "time_step": int(self.current_time),
            "active_ue_ids": [int(device["id"]) for device in self.active_devices],
            "connected_ue_ids": [
                int(device["id"])
                for device in self.active_devices
                if device.get("connected_bs") is not None
            ],
            "active_ue_count": int(active_mask.sum()),
            "connected_ue_count": int(connected_mask.sum()),
            "active_mask": active_mask,
            "connected_mask": connected_mask,
            "slot_ue_ids": np.asarray(slot_ue_ids, dtype=np.int64),
            "throughput": (
                self._calculate_throughput() if throughput is None else float(throughput)
            ),
            "fairness": self._calculate_fairness() if fairness is None else float(fairness),
            "blocking": self._calculate_blocking() if blocking is None else float(blocking),
            "reward": reward,
            "load_balance": float(-np.std(loads)) if loads else 0.0,
            "cell_loads": np.asarray(loads, dtype=np.float32),
            "offered_loads": np.asarray(loads, dtype=np.float32),
            "served_utilizations": np.asarray(
                [bs["served_utilization"] for bs in self.base_stations],
                dtype=np.float32,
            ),
            "required_prbs": np.asarray(
                [bs["required_prbs"] for bs in self.base_stations],
                dtype=np.float32,
            ),
            "pre_action_loads": (
                None
                if pre_action_loads is None
                else np.asarray(pre_action_loads, dtype=np.float32)
            ),
            "handover_count": int(self.last_handover_count),
            "measurement_reassociation_count": int(self.last_measurement_reassociation_count),
            "pingpong_count": int(self.last_pingpong_count),
            "cio_change": float(self.last_cio_change_magnitude),
            "total_prbs": int(self._total_prbs()),
            "serving_cell_ids": self._get_serving_cell_ids(),
            "rsrp_matrix": self._get_rsrp_matrix(),
            "measurement_mask": self._get_measurement_mask_observation().reshape(
                self.num_base_stations, self.num_devices
            ),
            "positions": {
                int(device["id"]): list(device["current_position"])
                for device in self.active_devices
            },
            "cio_matrix": {
                f"{source}->{target}": float(value)
                for (source, target), value in self.cio_matrix.items()
            },
        }

    def _record_visualization_metrics(
        self, throughput=None, fairness=None, blocking=None
    ):
        if throughput is None:
            throughput = self._calculate_throughput()
        if fairness is None:
            fairness = self._calculate_fairness()
        if blocking is None:
            blocking = self._calculate_blocking()

        loads = self._get_bs_loads()
        load_metric = float(np.std(loads))
        self.throughput_history.append(float(throughput))
        self.fairness_history.append(float(fairness))
        self.blocking_history.append(float(blocking))
        self.average_load_history.append(load_metric)
        # Record each simulation step, including steps that are not rendered.
        self.population_history.append((self.current_time, len(self.active_devices)))
        present_counts = self.config.get("present_user_count_by_step")
        present = present_counts[self.current_time] if present_counts is not None else len(self.active_devices)
        self.present_population_history.append((self.current_time, present))

    def _record_active_positions(self):
        for device in self.active_devices:
            trajectory = self.device_trajectories.setdefault(
                int(device["id"]), deque(maxlen=50)
            )
            trajectory.append(list(device["current_position"]))

    def _load_osm_map(self):
        """Load OSM map data and convert it to the environment coordinate system."""
        if self.map_loaded or not self.osm_file_path or not OSM_AVAILABLE:
            return

        try:
            print("Loading OSM map data...")
            self.osm_graph = ox.graph_from_xml(self.osm_file_path)
            self.road_nodes, self.road_edges = ox.graph_to_gdfs(self.osm_graph)
            self._convert_osm_to_meters()
            self.map_loaded = True
            print(
                f"OSM map loaded: {len(self.road_nodes)} road nodes, "
                f"{len(self.road_edges)} road edges"
            )
        except Exception as error:
            print(f"Error loading OSM map: {error}")
            self.map_loaded = False

    def _convert_osm_to_meters(self):
        """Convert OSM latitude/longitude coordinates to metric coordinates."""
        min_lon = self.road_nodes.geometry.x.min()
        min_lat = self.road_nodes.geometry.y.min()
        max_lat = self.road_nodes.geometry.y.max()
        center_lat = (min_lat + max_lat) / 2
        earth_radius = 6371000.0

        def lon_to_meters(lon):
            lon_diff_rad = math.radians(lon - min_lon)
            return lon_diff_rad * earth_radius * math.cos(math.radians(center_lat))

        def lat_to_meters(lat):
            lat_diff_rad = math.radians(lat - min_lat)
            return lat_diff_rad * earth_radius

        from shapely.geometry import LineString, Point

        for node_id, node_data in self.road_nodes.iterrows():
            lon = node_data.geometry.x
            lat = node_data.geometry.y
            self.road_nodes.at[node_id, "geometry"] = Point(
                lon_to_meters(lon), lat_to_meters(lat)
            )

        for edge_id, edge_data in self.road_edges.iterrows():
            if hasattr(edge_data.geometry, "coords"):
                converted_coords = [
                    (lon_to_meters(lon), lat_to_meters(lat))
                    for lon, lat in edge_data.geometry.coords
                ]
                self.road_edges.at[edge_id, "geometry"] = LineString(
                    converted_coords
                )

        print("OSM map coordinates converted to metric coordinate system")

    def render(self, mode="human"):
        is_real_scenario = bool(self.osm_file_path)
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            aspect_ratio = self.scene_width / self.scene_height
            base_height = 10
            if is_real_scenario:
                calculated_width = base_height * aspect_ratio
            else:
                calculated_width = (base_height * aspect_ratio) / 0.75
            max_width = 20
            min_width = 8
            fig_width = max(min(calculated_width, max_width), min_width)

            self.fig, self.ax = plt.subplots(figsize=(fig_width, base_height))
            population_bounds = (
                [0.71, 0.12, 0.24, 0.14]
                if is_real_scenario else [0.79, 0.23, 0.18, 0.17]
            )
            self.population_ax = self.fig.add_axes(population_bounds)
            self._sidebar_texts = []
            self.rendering = True

        plt.figure(self.fig.number)
        self.ax.clear()
        for text in self._sidebar_texts:
            text.remove()
        self._sidebar_texts.clear()

        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["mathtext.fontset"] = "stix"

        if is_real_scenario:
            if not self.map_loaded:
                self._load_osm_map()
            if self.map_loaded:
                for _, edge in self.road_edges.iterrows():
                    if hasattr(edge.geometry, "coords"):
                        coords = list(edge.geometry.coords)
                        x_coords = [coord[0] for coord in coords]
                        y_coords = [coord[1] for coord in coords]
                        self.ax.plot(
                            x_coords,
                            y_coords,
                            "gray",
                            linewidth=0.5,
                            alpha=0.7,
                        )

        self.ax.set_xlim(0, self.scene_width)
        self.ax.set_ylim(0, self.scene_height)
        self.ax.set_aspect("equal")
        title = (
            f"Map Based Scenario CIO Load Balancing - Step: {self.current_time}"
            if is_real_scenario
            else f"CIO Load Balancing Environment - Time: {self.current_time}"
        )
        self.ax.set_title(title, fontsize=16)

        for bs in self.base_stations:
            self.ax.plot(
                bs["position"][0],
                bs["position"][1],
                "s",
                color="red",
                markersize=12,
                markeredgecolor="darkred",
                markeredgewidth=2,
            )
            bs_label = (
                f"BS{bs['id']}"
                if is_real_scenario
                else f"BS{bs['id']}\nLoad:{bs['resource_utilization']:.2f}"
            )
            self.ax.text(
                bs["position"][0],
                bs["position"][1] + 0.5,
                bs_label,
                ha="center",
                va="bottom",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"),
            )

        colors = [
            "blue",
            "orange",
            "purple",
            "brown",
            "pink",
            "olive",
            "cyan",
            "magenta",
        ]
        for device in self.active_devices:
            device_id = int(device["id"])
            trajectory = self.device_trajectories.get(device_id, ())
            if len(trajectory) > 1:
                x_coords = [position[0] for position in trajectory]
                y_coords = [position[1] for position in trajectory]
                color = colors[int(device["slot"]) % len(colors)]
                self.ax.plot(
                    x_coords, y_coords, "-", color=color, alpha=0.7, linewidth=2
                )

        for device in self.active_devices:
            color = "green" if device["connected_bs"] is not None else "gray"
            self.ax.plot(
                device["current_position"][0],
                device["current_position"][1],
                "o",
                color=color,
                markersize=10,
                markeredgecolor="black",
                markeredgewidth=1,
            )
            if not is_real_scenario:
                self.ax.text(
                    device["current_position"][0],
                    device["current_position"][1] + 0.5,
                    f"D{device['id']}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    bbox=dict(
                        facecolor="white", alpha=0.8, boxstyle="round,pad=0.2"
                    ),
                )

        for device in self.active_devices:
            if device["connected_bs"] is not None:
                bs_position = device["connected_bs"]["position"]
                device_position = device["current_position"]
                self.ax.plot(
                    [bs_position[0], device_position[0]],
                    [bs_position[1], device_position[1]],
                    color="blue",
                    alpha=0.4,
                    linewidth=1,
                )

        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="red",
                markersize=10,
                label="Base Station",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="green",
                markersize=8,
                label="User Devices",
            ),
        ]

        current_throughput = (
            self.throughput_history[-1] if self.throughput_history else 0
        ) / 1000.0
        current_fairness = self.fairness_history[-1] if self.fairness_history else 0
        current_blocking = self.blocking_history[-1] if self.blocking_history else 0
        current_average_load = (
            self.average_load_history[-1] if self.average_load_history else 0
        )
        current_active_users = len(self.active_devices)

        if is_real_scenario:
            self.fig.subplots_adjust(right=0.64)
            load_info = "Loads:\n"
            for bs in self.base_stations:
                load_info += f"BS{bs['id']}: {bs['resource_utilization']:.2f}\n"
            performance_text = (
                "Performance Metrics:\n"
                f"Load Std: {current_average_load:.2f}\n"
                f"\n{load_info}"
            )
            text_x, performance_y, cio_y = 0.68, 0.70, 0.40
        else:
            self.fig.subplots_adjust(left=0.05, bottom=0.05, top=0.95, right=0.75)
            performance_text = (
                f"Active Users: {current_active_users}\n"
                f"Throughput: {current_throughput:.2f} kbps\n"
                f"Fairness: {current_fairness:.2f}\n"
                f"Blocking: {current_blocking:.2f}\n"
                f"Load Std: {current_average_load:.2f}\n"
            )
            text_x, performance_y, cio_y = 0.76, 0.75, 0.60

        self._sidebar_texts.append(self.fig.text(
            text_x,
            performance_y,
            performance_text,
            fontsize=11,
            bbox=dict(facecolor="white", alpha=0.9, boxstyle="round,pad=0.5"),
            verticalalignment="top",
            horizontalalignment="left",
            linespacing=1.4,
        ))

        bs_ids = sorted([bs["id"] for bs in self.base_stations])
        cio_matrix_str = "CIO Matrix:\n"
        cio_matrix_str += "     " + " ".join(
            [f"BS{bs_id:>4}" for bs_id in bs_ids]
        ) + "\n"
        for i, source_id in enumerate(bs_ids):
            row = f"BS{source_id} "
            for j, target_id in enumerate(bs_ids):
                value = (
                    0.0
                    if i == j
                    else self.cio_matrix.get((source_id, target_id), 0.0)
                )
                row += f"{value:6.2f} "
            cio_matrix_str += row + "\n"

        self._sidebar_texts.append(self.fig.text(
            text_x,
            cio_y,
            cio_matrix_str,
            fontsize=9,
            bbox=dict(facecolor="lightblue", alpha=0.8, boxstyle="round,pad=0.5"),
            verticalalignment="top",
            horizontalalignment="left",
            linespacing=1.2,
            fontfamily="monospace",
        ))

        self.population_ax.clear()
        steps, counts = zip(*self.population_history)
        self.population_ax.plot(steps, counts, color="green", linewidth=1.5)
        self.population_ax.plot(steps[-1], counts[-1], "o", color="green", markersize=4)
        self.population_ax.set_title("Active Users", fontsize=11)
        if "present_user_count_by_step" in self.config:
            _, present_counts = zip(*self.present_population_history)
            self.population_ax.lines[0].set_label("Online")
            self.population_ax.plot(steps, present_counts, color="gray", linestyle="--",
                                    linewidth=1.2, label="Present")
            self.population_ax.legend(fontsize=7, loc="best", framealpha=0.8)
            self.population_ax.set_title("Population", fontsize=11)
        else:
            present_counts = counts
        self.population_ax.set_xlabel("Step", fontsize=9)
        self.population_ax.set_ylabel("Count", fontsize=9)
        self.population_ax.set_xlim(0, max(1, self.current_time))
        self.population_ax.set_ylim(0, max(1, max(present_counts) * 1.15))
        from matplotlib.ticker import MaxNLocator

        self.population_ax.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        self.population_ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        self.population_ax.tick_params(labelsize=8)
        self.population_ax.grid(True, color="gray", alpha=0.25, linewidth=0.5)

        legend_kwargs = {}
        if is_real_scenario:
            legend_kwargs["bbox_to_anchor"] = (1.00, 1.00)
        self.ax.legend(
            handles=legend_elements,
            loc="upper left",
            fancybox=True,
            shadow=True,
            fontsize=10,
            ncol=1,
            **legend_kwargs,
        )

        self.fig.canvas.draw_idle()
        plt.pause(0.01)

        if self.save_animation_path:
            self.fig.canvas.draw()
            frame = np.array(self.fig.canvas.buffer_rgba())
            self.frames.append(frame)

        if mode == "human":
            try:
                input("Press Enter to continue...")
            except (EOFError, KeyboardInterrupt):
                pass

    def close(self):
        if self.save_animation_path and self.frames:
            self._save_animation()

        if self.fig is not None:
            plt.close(self.fig)
        self.fig = None
        self.ax = None
        self.population_ax = None
        self._sidebar_texts.clear()
        self.frames.clear()
        self.rendering = False

    def _save_animation(self):
        """Save captured visualization frames as a GIF."""
        if not self.frames:
            print("No frames to save")
            return

        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.save_animation_path)), exist_ok=True)
            from PIL import Image

            pil_images = [Image.fromarray(frame).convert("RGB") for frame in self.frames]
            if pil_images:
                pil_images[0].save(
                    self.save_animation_path,
                    save_all=True,
                    append_images=pil_images[1:],
                    duration=200,
                    loop=0,
                )
                print(f"Animation saved to: {self.save_animation_path}")
        except Exception as error:
            print(f"Error saving animation: {error}")

    def save_animation_at_end(self):
        """Save the accumulated animation frames, if configured."""
        if self.save_animation_path and self.frames:
            self._save_animation()

    def __call__(self):
        return self
