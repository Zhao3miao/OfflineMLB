import copy
from pathlib import Path

import yaml

from .scenario_lifecycle import split_disconnected_lifecycles


def load_scenario_config(config_or_path):
    if isinstance(config_or_path, (str, Path)):
        with open(config_or_path, "r", encoding="utf-8") as f:
            return yaml.load(f, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    return copy.deepcopy(config_or_path)


class Scenario:
    """Runtime view of a UE lifecycle scenario."""

    def __init__(self, config):
        self.config = config
        self.scene = config["scene"]
        self.total_steps = int(self.scene["total_steps"])
        # Snapshot cadence and control cadence are separate knobs. Snapshots are
        # the observation grid; control may be coarser, in which case the
        # environment advances one control interval per step and only evaluates
        # the snapshots that are control points. With no explicit configuration
        # the control interval equals the snapshot interval, which reproduces the
        # historical one-snapshot-per-step behaviour.
        self.snapshot_seconds = float(
            self.scene.get("snapshot_seconds", self.scene.get("step_seconds", 1.0)))
        self.simulation_seconds = float(self.scene.get("simulation_seconds", self.snapshot_seconds))
        self.control_seconds = float(self.scene.get("control_seconds", self.snapshot_seconds))
        if self.snapshot_seconds <= 0 or self.control_seconds <= 0:
            raise ValueError("snapshot_seconds and control_seconds must be positive")
        ratio = self.control_seconds / self.snapshot_seconds
        if abs(ratio - round(ratio)) > 1e-9 or round(ratio) < 1:
            raise ValueError("control_seconds must be an integer multiple of snapshot_seconds")
        self.control_stride = int(round(ratio))
        # Control steps until the episode ends; the final interval may be
        # shortened when the stride does not divide the horizon evenly.
        self.control_steps = -(-(self.total_steps - 1) // self.control_stride)
        self.base_stations = copy.deepcopy(config["base_stations"])
        self.max_active_users = int(config.get("max_active_users", 0))

        if "ue_lifecycles" not in config:
            raise ValueError("Scenario config must contain ue_lifecycles")
        lifecycles = split_disconnected_lifecycles(config["ue_lifecycles"])

        self.lifecycles = [self._prepare_lifecycle(item) for item in lifecycles]
        if self.max_active_users <= 0:
            self.max_active_users = self._infer_max_active_users()

    def active_users_at(self, step):
        users = []
        for lifecycle in self.lifecycles:
            if not (lifecycle["spawn_step"] <= step < lifecycle["leave_step"]):
                continue
            position = lifecycle["_position_by_step"].get(step)
            if position is None:
                continue
            users.append(
                {
                    "ue_id": lifecycle["ue_id"],
                    "position": position,
                    "initial_base_station_id": lifecycle.get("initial_base_station_id"),
                    "connected_allowed": True,
                    "lifecycle": lifecycle,
                }
            )
        users.sort(key=lambda item: item["ue_id"])
        return users

    def _prepare_lifecycle(self, lifecycle):
        item = copy.deepcopy(lifecycle)
        if "ue_id" not in item:
            raise ValueError("Each UE lifecycle must define ue_id")
        item["ue_id"] = int(item["ue_id"])
        item["spawn_step"] = int(item.get("spawn_step", 0))
        item["leave_step"] = int(item.get("leave_step", self.total_steps))
        item.setdefault("disconnect_intervals", [])

        position_by_step = {}
        for point in item.get("trajectory", []):
            step = int(point["step"])
            position_by_step[step] = [
                float(point["position"][0]),
                float(point["position"][1]),
            ]
        item["_position_by_step"] = position_by_step
        return item

    def _infer_max_active_users(self):
        max_active = 0
        for step in range(self.total_steps + 1):
            max_active = max(max_active, len(self.active_users_at(step)))
        return max_active
