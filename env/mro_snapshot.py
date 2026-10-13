import json
from pathlib import Path

import numpy as np


MRO_SNAPSHOT_FORMAT = "mro_snapshot_v1"
SNAPSHOT_FILENAME = "snapshots.npz"
MANIFEST_FILENAME = "manifest.json"


def load_mro_snapshot_manifest(processed_dir):
    processed_dir = Path(processed_dir)
    manifest_path = processed_dir / MANIFEST_FILENAME
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format_version") != MRO_SNAPSHOT_FORMAT:
        raise ValueError(
            "Expected MRO snapshot format %s, got %s"
            % (MRO_SNAPSHOT_FORMAT, manifest.get("format_version"))
        )
    return manifest


class MROSnapshotScenario:
    """Runtime scenario backed by the supplied anonymized measurement snapshots."""

    def __init__(self, processed_dir):
        self.processed_dir = Path(processed_dir)
        self.manifest = load_mro_snapshot_manifest(self.processed_dir)
        snapshot_path = self.processed_dir / SNAPSHOT_FILENAME
        with np.load(str(snapshot_path), allow_pickle=False) as arrays:
            self.timestamps = np.asarray(arrays["timestamps"]).astype(str)
            self.step_offsets = np.asarray(arrays["step_offsets"], dtype=np.int64)
            self.ue_ids = np.asarray(arrays["ue_ids"], dtype=np.int64)
            self.rsrp = np.asarray(arrays["rsrp"], dtype=np.float32)
            self.measurement_mask = np.asarray(
                arrays["measurement_mask"], dtype=np.bool_
            )
            self.initial_association = np.asarray(
                arrays["initial_association"], dtype=np.int16
            )

        self._validate_arrays()
        self.total_steps = len(self.timestamps)
        self.max_active_users = int(self.manifest["statistics"]["max_active_users"])
        self.base_stations = self._build_base_stations()
        self.lifecycles = []
        self.config = self._build_environment_config()
        self.scene = self.config["scene"]

    def _validate_arrays(self):
        num_records = len(self.ue_ids)
        num_cells = len(self.manifest["target_cells"])
        if self.step_offsets.shape != (len(self.timestamps) + 1,):
            raise ValueError("step_offsets must have len(timestamps) + 1 entries")
        if self.step_offsets[0] != 0 or self.step_offsets[-1] != num_records:
            raise ValueError("step_offsets do not cover all snapshot records")
        if np.any(np.diff(self.step_offsets) < 0):
            raise ValueError("step_offsets must be non-decreasing")
        if self.rsrp.shape != (num_records, num_cells):
            raise ValueError("rsrp array shape does not match records and target cells")
        if self.measurement_mask.shape != self.rsrp.shape:
            raise ValueError("measurement_mask shape must match rsrp shape")
        if np.any(self.measurement_mask.sum(axis=1) == 0):
            raise ValueError("All-zero target-cell measurement rows are not allowed")
        if len(self.initial_association) != num_records:
            raise ValueError("initial_association length does not match ue_ids")
        if np.any(self.initial_association < 1) or np.any(
            self.initial_association > num_cells
        ):
            raise ValueError("initial_association contains an invalid local cell id")

    def _build_base_stations(self):
        num_cells = len(self.manifest["target_cells"])
        radius = 400.0
        center = np.asarray([500.0, 500.0])
        base_stations = []
        for index, cell in enumerate(self.manifest["target_cells"]):
            angle = 2.0 * np.pi * index / max(num_cells, 1)
            position = center + radius * np.asarray([np.cos(angle), np.sin(angle)])
            base_stations.append(
                {
                    "id": index + 1,
                    "position": position.tolist(),
                    "tx_power": 46,
                }
            )
        return base_stations

    def _build_environment_config(self):
        return {
            "schema_version": MRO_SNAPSHOT_FORMAT,
            "scenario_id": self.manifest["scenario_id"],
            "scene": {
                "width": 1000,
                "height": 1000,
                "total_steps": self.total_steps,
            },
            "base_stations": self.base_stations,
            "max_active_users": self.max_active_users,
            "observation_config": {
                "type": "factored",
                "components": [
                    "rsrp",
                    "cio",
                    "association",
                    "measurement_mask",
                    "active_mask",
                ],
                "inactive_rsrp": float(self.manifest["missing_rsrp_fill"]),
            },
            "action_config": {"mode": "delta", "delta_scale": 0.5},
            "cio_config": {"min_value": -6, "max_value": 6, "step_size": 0.5},
            "hysteresis_config": {"enabled": True, "value": 2.0},
            "reward_config": {"type": "load_balance"},
        }

    def active_users_at(self, step):
        if step < 0 or step >= self.total_steps:
            return []
        start = int(self.step_offsets[step])
        end = int(self.step_offsets[step + 1])
        users = []
        for record_index in range(start, end):
            measured_indices = np.flatnonzero(self.measurement_mask[record_index])
            measurements = {
                int(cell_index + 1): float(self.rsrp[record_index, cell_index])
                for cell_index in measured_indices
            }
            mask = {
                int(cell_index + 1): bool(
                    self.measurement_mask[record_index, cell_index]
                )
                for cell_index in range(self.measurement_mask.shape[1])
            }
            initial_bs = int(self.initial_association[record_index])
            users.append(
                {
                    "ue_id": int(self.ue_ids[record_index]),
                    "position": list(self.base_stations[initial_bs - 1]["position"]),
                    "initial_base_station_id": initial_bs,
                    "connected_allowed": True,
                    "rsrp_measurements": measurements,
                    "measurement_mask": mask,
                }
            )
        return users
