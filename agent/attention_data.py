from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import Dataset


NORMALIZATION_VERSION = "fixed_minmax_count_v1"
DATASET_SCHEMA_VERSION = "attention_set_v4"
SUPPORTED_DATASET_SCHEMA_VERSIONS = {"attention_set_v3", DATASET_SCHEMA_VERSION}
DEFAULT_RSRP_MIN = -120.0
DEFAULT_RSRP_MAX = -50.0
DEFAULT_USER_COUNT_CAP = 100.0


def observation_to_attention_inputs(environment, observation):
    """Convert the factored fixed-slot observation into a ragged UE set."""
    parts = environment._split_observation(observation)
    required = {"rsrp", "cio", "association", "active_mask"}
    missing = required.difference(parts)
    if missing:
        raise ValueError("Attention observation is missing components: %s" % sorted(missing))

    num_bs = environment.num_base_stations
    num_slots = environment.num_devices
    rsrp = parts["rsrp"].reshape(num_bs, num_slots).T
    association = parts["association"].reshape(num_bs, num_slots).T
    active = parts["active_mask"].astype(bool)
    feature_parts = [rsrp]
    if "measurement_mask" in parts:
        feature_parts.append(
            parts["measurement_mask"].reshape(num_bs, num_slots).T
        )
    feature_parts.append(association)
    user_features = np.concatenate(feature_parts, axis=1)
    global_features = np.concatenate(
        [parts["cio"], np.asarray([active.sum()], dtype=np.float32)]
    )
    return user_features[active].astype(np.float32), global_features.astype(np.float32)


def token_feature_names(num_base_stations, include_measurement_mask=False):
    names = ["rsrp_bs_%d" % (index + 1) for index in range(num_base_stations)]
    if include_measurement_mask:
        names += [
            "measured_bs_%d" % (index + 1)
            for index in range(num_base_stations)
        ]
    return names + [
        "associated_bs_%d" % (index + 1)
        for index in range(num_base_stations)
    ]


def make_fixed_normalization(
    num_base_stations,
    rsrp_min=DEFAULT_RSRP_MIN,
    rsrp_max=DEFAULT_RSRP_MAX,
    user_count_cap=DEFAULT_USER_COUNT_CAP,
):
    if rsrp_max <= rsrp_min:
        raise ValueError("rsrp_max must be greater than rsrp_min")
    if user_count_cap <= 0:
        raise ValueError("user_count_cap must be positive")
    return {
        "version": NORMALIZATION_VERSION,
        "num_base_stations": int(num_base_stations),
        "rsrp_min": float(rsrp_min),
        "rsrp_max": float(rsrp_max),
        "user_count_cap": float(user_count_cap),
    }


def _object_array_to_list(values):
    return [np.asarray(value, dtype=np.float32) for value in values]


def _restore_trajectory_ids(time_step, terminal):
    """Recover contiguous episode ids used by datasets created before schema v4."""
    trajectory_ids = np.empty(len(time_step), dtype=np.int64)
    trajectory_id = -1
    previous_terminal = True
    for index, step in enumerate(time_step):
        if previous_terminal or int(step) == 0:
            trajectory_id += 1
        trajectory_ids[index] = trajectory_id
        previous_terminal = bool(terminal[index])
    return trajectory_ids.astype(str)


def load_attention_dataset(path):
    path = Path(path)
    if path.suffix == ".json":
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("format") != "attention_dataset_parts_v1" or not manifest.get("parts"):
            raise ValueError("Invalid attention dataset manifest: %s" % path)
        parts = [load_attention_dataset(path.parent / name) for name in manifest["parts"]]
        normalization = parts[0]["normalization"]
        if any(part["normalization"] != normalization for part in parts):
            raise ValueError("Dataset parts use different normalization")
        held_out = set(manifest.get("excluded_scenario_ids", []))
        seen_scenarios = set()
        for part in parts:
            ids = set(part["scenario_id"])
            if ids & held_out:
                raise ValueError("Held-out scenario leakage in dataset manifest")
            if ids & seen_scenarios:
                raise ValueError("Scenario repeated in different dataset parts")
            seen_scenarios.update(ids)
        result = {"normalization": normalization, "dataset_schema_version": parts[0]["dataset_schema_version"]}
        for key in parts[0]:
            if key in result:
                continue
            if isinstance(parts[0][key], list):
                result[key] = [item for part in parts for item in part[key]]
            else:
                result[key] = np.concatenate([part[key] for part in parts], axis=0)
        return result
    with np.load(str(path), allow_pickle=True) as data:
        required = {
            "user_obs",
            "next_user_obs",
            "global_obs",
            "next_global_obs",
            "action",
            "reward",
            "terminal",
            "dataset_schema_version",
            "normalization_version",
            "num_base_stations",
            "rsrp_min",
            "rsrp_max",
            "user_count_cap",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError("Dataset %s is missing arrays: %s" % (path, sorted(missing)))
        dataset_schema_version = str(data["dataset_schema_version"].item())
        if dataset_schema_version not in SUPPORTED_DATASET_SCHEMA_VERSIONS:
            raise ValueError(
                "Expected one of dataset schemas %s, got %s"
                % (sorted(SUPPORTED_DATASET_SCHEMA_VERSIONS), dataset_schema_version)
            )
        transition_count = len(data["action"])
        scenario_id = (
            np.asarray(data["scenario_id"]).astype(str)
            if "scenario_id" in data.files
            else np.repeat("unknown", transition_count)
        )
        time_step = (
            np.asarray(data["time_step"], dtype=np.int64)
            if "time_step" in data.files
            else np.arange(transition_count, dtype=np.int64)
        )
        terminal = np.asarray(data["terminal"], dtype=np.float32)
        trajectory_id = (
            np.asarray(data["trajectory_id"]).astype(str)
            if "trajectory_id" in data.files
            else _restore_trajectory_ids(time_step, terminal)
        )
        return {
            "user_obs": _object_array_to_list(data["user_obs"]),
            "next_user_obs": _object_array_to_list(data["next_user_obs"]),
            "global_obs": np.asarray(data["global_obs"], dtype=np.float32),
            "next_global_obs": np.asarray(data["next_global_obs"], dtype=np.float32),
            "action": np.asarray(data["action"], dtype=np.float32),
            "reward": np.asarray(data["reward"], dtype=np.float32),
            "terminal": terminal,
            "scenario_id": scenario_id,
            "time_step": time_step,
            "trajectory_id": trajectory_id,
            "normalization": {
                "version": str(data["normalization_version"].item()),
                "num_base_stations": int(data["num_base_stations"].item()),
                "rsrp_min": float(data["rsrp_min"].item()),
                "rsrp_max": float(data["rsrp_max"].item()),
                "user_count_cap": float(data["user_count_cap"].item()),
            },
            "dataset_schema_version": dataset_schema_version,
        }


TRAIN_ZSCORE_VERSION = "train_zscore_count_v1"


def add_rsrp_normalization_args(parser):
    parser.add_argument("--rsrp-normalization", choices=("fixed", "train_zscore"), default="fixed",
                        help="Fit one shared RSRP mean/std on valid training measurements only")


def training_normalization(arrays, mode="fixed"):
    """Pool current and next training observations, weighted per valid measurement.

    Ragged inputs contain active users only. Measurement masks exclude missing
    cells. Both transition endpoints participate, including terminal snapshots;
    repeated observations retain their offline dataset frequency. No test data
    or sampled minibatches enter the fit. The source arrays remain unchanged.
    """
    normalization = dict(arrays["normalization"])
    if mode == "fixed":
        return normalization
    if mode != "train_zscore":
        raise ValueError("Unknown RSRP normalization: %s" % mode)
    num_bs = int(normalization["num_base_stations"])
    count, mean, m2 = 0, 0.0, 0.0
    for key in ("user_obs", "next_user_obs"):
        for tokens in arrays[key]:
            tokens = np.asarray(tokens)
            if tokens.ndim != 2 or tokens.shape[1] not in (2 * num_bs, 3 * num_bs):
                raise ValueError("Invalid training token shape")
            valid = (tokens[:, num_bs:2 * num_bs] > 0 if tokens.shape[1] == 3 * num_bs
                     else np.ones(tokens[:, :num_bs].shape, dtype=bool))
            values = tokens[:, :num_bs][valid].astype(np.float64)
            if not values.size:
                continue
            if not np.isfinite(values).all():
                raise ValueError("Non-finite measured training RSRP")
            size, batch_mean = values.size, float(values.mean())
            delta, total = batch_mean - mean, count + size
            m2 += float(np.square(values - batch_mean).sum()) + delta * delta * count * size / total
            mean += delta * size / total
            count = total
    if not count:
        raise ValueError("No valid training RSRP measurements")
    std = float(np.sqrt(m2 / count))
    if not np.isfinite(std) or std <= 1e-8:
        raise ValueError("Training RSRP standard deviation is zero or invalid")
    normalization.update(version=TRAIN_ZSCORE_VERSION, rsrp_mean=float(mean), rsrp_std=std,
                         rsrp_valid_count=int(count), rsrp_std_ddof=0,
                         rsrp_fit_source="training_transition_current_and_next_valid_measurements",
                         rsrp_pooling="shared_across_cells", rsrp_clip=False)
    return normalization


def _validate_normalization(normalization):
    if normalization.get("version") not in (NORMALIZATION_VERSION, TRAIN_ZSCORE_VERSION):
        raise ValueError(
            "Expected normalization version %s, got %s"
            % (NORMALIZATION_VERSION, normalization.get("version"))
        )


def normalize_tokens(values, normalization):
    _validate_normalization(normalization)
    values = np.asarray(values, dtype=np.float32).copy()
    num_bs = int(normalization["num_base_stations"])
    valid_dims = (2 * num_bs, 3 * num_bs)
    if values.ndim != 2 or values.shape[1] not in valid_dims:
        raise ValueError("Expected user token dimension in %s" % (valid_dims,))
    if normalization["version"] == TRAIN_ZSCORE_VERSION:
        mean, std = float(normalization["rsrp_mean"]), float(normalization["rsrp_std"])
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 1e-8:
            raise ValueError("Invalid saved RSRP mean/std")
        values[:, :num_bs] = (values[:, :num_bs] - mean) / std
        if values.shape[1] == 3 * num_bs:
            values[:, :num_bs] = np.where(values[:, num_bs:2 * num_bs] > 0,
                                           values[:, :num_bs], 0.0)
    else:
        rsrp_min = float(normalization["rsrp_min"])
        rsrp_max = float(normalization["rsrp_max"])
        values[:, :num_bs] = (
            np.clip(values[:, :num_bs], rsrp_min, rsrp_max) - rsrp_min
        ) / (rsrp_max - rsrp_min)
    values[:, num_bs:] = np.clip(values[:, num_bs:], 0.0, 1.0)
    return values


def normalize_global(values, normalization):
    _validate_normalization(normalization)
    values = np.asarray(values, dtype=np.float32).copy()
    num_bs = int(normalization["num_base_stations"])
    cio_dim = num_bs * (num_bs - 1)
    if values.ndim != 1 or values.shape[0] != cio_dim + 1:
        raise ValueError("Expected global observation dimension %d" % (cio_dim + 1))
    values[:cio_dim] = np.clip(values[:cio_dim], 0.0, 1.0)
    user_count_cap = float(normalization["user_count_cap"])
    if values[-1] < 0 or values[-1] > user_count_cap:
        raise ValueError(
            "Visible user count %.1f is outside fixed normalization range [0, %.1f]"
            % (values[-1], user_count_cap)
        )
    values[-1] = values[-1] / user_count_cap
    return values


class AttentionTransitionDataset(Dataset):
    def __init__(self, arrays, normalization):
        self.arrays = arrays
        self.normalization = normalization

    def __len__(self):
        return len(self.arrays["action"])

    def _normalize_tokens(self, values):
        return normalize_tokens(values, self.normalization)

    def _normalize_global(self, values):
        return normalize_global(values, self.normalization)

    def __getitem__(self, index):
        return {
            "user_obs": self._normalize_tokens(self.arrays["user_obs"][index]),
            "next_user_obs": self._normalize_tokens(self.arrays["next_user_obs"][index]),
            "global_obs": self._normalize_global(self.arrays["global_obs"][index]),
            "next_global_obs": self._normalize_global(self.arrays["next_global_obs"][index]),
            "action": self.arrays["action"][index],
            "reward": self.arrays["reward"][index],
            "terminal": self.arrays["terminal"][index],
        }


def _pad_sets(values):
    batch_size = len(values)
    feature_dim = values[0].shape[1]
    max_users = max(1, max(value.shape[0] for value in values))
    padded = np.zeros((batch_size, max_users, feature_dim), dtype=np.float32)
    mask = np.zeros((batch_size, max_users), dtype=np.bool_)
    for index, value in enumerate(values):
        count = value.shape[0]
        if count:
            padded[index, :count] = value
            mask[index, :count] = True
    return torch.from_numpy(padded), torch.from_numpy(mask)


def collate_attention_transitions(batch):
    user_obs, user_mask = _pad_sets([item["user_obs"] for item in batch])
    next_user_obs, next_user_mask = _pad_sets([item["next_user_obs"] for item in batch])
    return {
        "user_obs": user_obs,
        "user_mask": user_mask,
        "next_user_obs": next_user_obs,
        "next_user_mask": next_user_mask,
        "global_obs": torch.from_numpy(np.stack([item["global_obs"] for item in batch])),
        "next_global_obs": torch.from_numpy(np.stack([item["next_global_obs"] for item in batch])),
        "action": torch.from_numpy(np.stack([item["action"] for item in batch])),
        "reward": torch.tensor([item["reward"] for item in batch], dtype=torch.float32).unsqueeze(-1),
        "terminal": torch.tensor([item["terminal"] for item in batch], dtype=torch.float32).unsqueeze(-1),
    }


def normalize_single(user_obs, global_obs, normalization):
    user_obs = normalize_tokens(user_obs, normalization)
    global_obs = normalize_global(global_obs, normalization)
    tokens, mask = _pad_sets([user_obs])
    return tokens, mask, torch.from_numpy(global_obs).unsqueeze(0)
