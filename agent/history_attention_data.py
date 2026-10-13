import math
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from agent.attention_data import normalize_global, normalize_tokens


class HistoryAttentionTransitionDataset(Dataset):
    """Build fixed-length state histories without crossing trajectory boundaries."""

    def __init__(self, arrays, normalization, history_length=5):
        if history_length <= 0:
            raise ValueError("history_length must be positive")
        self.arrays = arrays
        self.normalization = normalization
        self.history_length = int(history_length)
        self.token_dim = int(arrays["user_obs"][0].shape[1])
        self.global_dim = int(arrays["global_obs"].shape[1])

        trajectory_indices = defaultdict(list)
        for index, trajectory_id in enumerate(arrays["trajectory_id"]):
            trajectory_indices[str(trajectory_id)].append(index)
        self.trajectory_indices = dict(trajectory_indices)
        self.position_in_trajectory = np.empty(len(self), dtype=np.int64)
        for indices in self.trajectory_indices.values():
            previous_step = None
            for position, index in enumerate(indices):
                step = int(arrays["time_step"][index])
                if previous_step is not None and step != previous_step + 1:
                    raise ValueError(
                        "Non-contiguous time steps within trajectory %s" % indices
                    )
                self.position_in_trajectory[index] = position
                previous_step = step

    def __len__(self):
        return len(self.arrays["action"])

    def _state(self, index):
        return (
            normalize_tokens(self.arrays["user_obs"][index], self.normalization),
            normalize_global(self.arrays["global_obs"][index], self.normalization),
        )

    def _next_state(self, index):
        return (
            normalize_tokens(
                self.arrays["next_user_obs"][index], self.normalization
            ),
            normalize_global(
                self.arrays["next_global_obs"][index], self.normalization
            ),
        )

    def _left_pad(self, states):
        padding = self.history_length - len(states)
        users = [np.zeros((0, self.token_dim), dtype=np.float32) for _ in range(padding)]
        globals_ = [np.zeros(self.global_dim, dtype=np.float32) for _ in range(padding)]
        users.extend(state[0] for state in states)
        globals_.extend(state[1] for state in states)
        valid = np.concatenate(
            [
                np.zeros(padding, dtype=np.bool_),
                np.ones(len(states), dtype=np.bool_),
            ]
        )
        return users, np.stack(globals_), valid

    def __getitem__(self, index):
        trajectory_id = str(self.arrays["trajectory_id"][index])
        indices = self.trajectory_indices[trajectory_id]
        position = int(self.position_in_trajectory[index])
        start = max(0, position - self.history_length + 1)
        current_states = [self._state(item) for item in indices[start : position + 1]]
        next_states = current_states + [self._next_state(index)]
        next_states = next_states[-self.history_length :]
        user_history, global_history, history_mask = self._left_pad(current_states)
        next_user_history, next_global_history, next_history_mask = self._left_pad(
            next_states
        )
        return {
            "index": int(index),
            "user_history": user_history,
            "global_history": global_history,
            "history_mask": history_mask,
            "next_user_history": next_user_history,
            "next_global_history": next_global_history,
            "next_history_mask": next_history_mask,
            "action": self.arrays["action"][index],
            "reward": self.arrays["reward"][index],
            "terminal": self.arrays["terminal"][index],
            "scenario_id": str(self.arrays["scenario_id"][index]),
            "time_step": int(self.arrays["time_step"][index]),
            "trajectory_id": trajectory_id,
        }


def _pad_history_sets(histories):
    batch_size = len(histories)
    history_length = len(histories[0])
    feature_dim = histories[0][-1].shape[1]
    max_users = max(
        1,
        max(frame.shape[0] for history in histories for frame in history),
    )
    padded = np.zeros(
        (batch_size, history_length, max_users, feature_dim), dtype=np.float32
    )
    mask = np.zeros((batch_size, history_length, max_users), dtype=np.bool_)
    for batch_index, history in enumerate(histories):
        for history_index, frame in enumerate(history):
            count = frame.shape[0]
            if count:
                padded[batch_index, history_index, :count] = frame
                mask[batch_index, history_index, :count] = True
    return torch.from_numpy(padded), torch.from_numpy(mask)


def collate_history_attention_transitions(batch):
    user_history, user_mask = _pad_history_sets(
        [item["user_history"] for item in batch]
    )
    next_user_history, next_user_mask = _pad_history_sets(
        [item["next_user_history"] for item in batch]
    )
    return {
        "index": torch.tensor([item["index"] for item in batch], dtype=torch.long),
        "user_history": user_history,
        "user_mask": user_mask,
        "global_history": torch.from_numpy(
            np.stack([item["global_history"] for item in batch])
        ),
        "history_mask": torch.from_numpy(
            np.stack([item["history_mask"] for item in batch])
        ),
        "next_user_history": next_user_history,
        "next_user_mask": next_user_mask,
        "next_global_history": torch.from_numpy(
            np.stack([item["next_global_history"] for item in batch])
        ),
        "next_history_mask": torch.from_numpy(
            np.stack([item["next_history_mask"] for item in batch])
        ),
        "action": torch.from_numpy(np.stack([item["action"] for item in batch])),
        "reward": torch.tensor(
            [item["reward"] for item in batch], dtype=torch.float32
        ).unsqueeze(-1),
        "terminal": torch.tensor(
            [item["terminal"] for item in batch], dtype=torch.float32
        ).unsqueeze(-1),
    }


class MatchedExogenousBatchSampler(Sampler):
    """Yield adjacent pairs sharing scenario/time but not trajectory."""

    def __init__(self, arrays, batch_size, seed=0):
        if batch_size <= 0 or batch_size % 2:
            raise ValueError("batch_size must be a positive even number")
        self.batch_size = int(batch_size)
        self.num_samples = len(arrays["action"])
        grouped = defaultdict(list)
        for index, (scenario_id, time_step) in enumerate(
            zip(arrays["scenario_id"], arrays["time_step"])
        ):
            grouped[(str(scenario_id), int(time_step))].append(index)
        self.groups = []
        for indices in grouped.values():
            trajectories = {str(arrays["trajectory_id"][index]) for index in indices}
            if len(trajectories) >= 2:
                self.groups.append(np.asarray(indices, dtype=np.int64))
        if not self.groups:
            raise ValueError(
                "No matched groups contain two or more distinct trajectories"
            )
        self.trajectory_id = np.asarray(arrays["trajectory_id"]).astype(str)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return max(1, math.ceil(self.num_samples / self.batch_size))

    def __iter__(self):
        pair_count = self.batch_size // 2
        for _ in range(len(self)):
            batch = []
            group_positions = self.rng.integers(0, len(self.groups), size=pair_count)
            for group_position in group_positions:
                group = self.groups[int(group_position)]
                first = int(self.rng.choice(group))
                candidates = group[self.trajectory_id[group] != self.trajectory_id[first]]
                second = int(self.rng.choice(candidates))
                batch.extend([first, second])
            yield batch


def normalize_history(history, normalization, history_length):
    """Convert an online deque of raw attention states into one history batch."""
    if not history:
        raise ValueError("history must contain at least one state")
    states = list(history)[-history_length:]
    token_dim = int(states[-1][0].shape[1])
    global_dim = int(states[-1][1].shape[0])
    padding = history_length - len(states)
    user_frames = [np.zeros((0, token_dim), dtype=np.float32) for _ in range(padding)]
    global_frames = [np.zeros(global_dim, dtype=np.float32) for _ in range(padding)]
    for user_obs, global_obs in states:
        user_frames.append(normalize_tokens(user_obs, normalization))
        global_frames.append(normalize_global(global_obs, normalization))
    history_mask = np.concatenate(
        [np.zeros(padding, dtype=np.bool_), np.ones(len(states), dtype=np.bool_)]
    )
    users, user_mask = _pad_history_sets([user_frames])
    return (
        users,
        user_mask,
        torch.from_numpy(np.stack([global_frames])),
        torch.from_numpy(history_mask).unsqueeze(0),
    )
