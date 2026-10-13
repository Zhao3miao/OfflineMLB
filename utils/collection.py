"""Offline transition storage using the original paper dataset schema."""
from pathlib import Path
import numpy as np
from agent.attention_data import DATASET_SCHEMA_VERSION, observation_to_attention_inputs
def empty_storage():
    return {
        "user_obs": [],
        "next_user_obs": [],
        "global_obs": [],
        "next_global_obs": [],
        "action": [],
        "reward": [],
        "terminal": [],
        "trajectory_id": [],
        "scenario_id": [],
        "time_step": [],
        "policy_name": [],
        "active_ue_count": [],
        "next_active_ue_count": [],
        "load_balance": [],
        "blocking": [],
        "handover_count": [],
    }


def append_transition(
    storage,
    environment,
    observation,
    next_observation,
    action,
    reward,
    terminal,
    transition_info,
    next_info,
    scenario_id,
    time_step,
    policy_name,
    trajectory_id=None,
):
    user_obs, global_obs = observation_to_attention_inputs(environment, observation)
    next_user_obs, next_global_obs = observation_to_attention_inputs(
        environment, next_observation
    )
    storage["user_obs"].append(user_obs)
    storage["next_user_obs"].append(next_user_obs)
    storage["global_obs"].append(global_obs)
    storage["next_global_obs"].append(next_global_obs)
    storage["action"].append(np.asarray(action, dtype=np.float32))
    storage["reward"].append(float(reward))
    storage["terminal"].append(bool(terminal))
    if trajectory_id is None:
        trajectory_id = scenario_id
    storage["trajectory_id"].append(str(trajectory_id))
    storage["scenario_id"].append(str(scenario_id))
    storage["time_step"].append(int(time_step))
    storage["policy_name"].append(str(policy_name))
    storage["active_ue_count"].append(int(user_obs.shape[0]))
    storage["next_active_ue_count"].append(int(next_user_obs.shape[0]))
    storage["load_balance"].append(float(transition_info["load_balance"]))
    storage["blocking"].append(float(transition_info["blocking"]))
    storage["handover_count"].append(int(transition_info["handover_count"]))


def as_object_array(values):
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def save_dataset(path, storage, normalization):
    np.savez_compressed(
        str(path),
        dataset_schema_version=np.asarray(DATASET_SCHEMA_VERSION),
        normalization_version=np.asarray(normalization["version"]),
        num_base_stations=np.asarray(normalization["num_base_stations"], dtype=np.int32),
        rsrp_min=np.asarray(normalization["rsrp_min"], dtype=np.float32),
        rsrp_max=np.asarray(normalization["rsrp_max"], dtype=np.float32),
        user_count_cap=np.asarray(normalization["user_count_cap"], dtype=np.float32),
        user_obs=as_object_array(storage["user_obs"]),
        next_user_obs=as_object_array(storage["next_user_obs"]),
        global_obs=np.asarray(storage["global_obs"], dtype=np.float32),
        next_global_obs=np.asarray(storage["next_global_obs"], dtype=np.float32),
        action=np.asarray(storage["action"], dtype=np.float32),
        reward=np.asarray(storage["reward"], dtype=np.float32),
        terminal=np.asarray(storage["terminal"], dtype=np.bool_),
        trajectory_id=np.asarray(storage["trajectory_id"]),
        scenario_id=np.asarray(storage["scenario_id"]),
        time_step=np.asarray(storage["time_step"], dtype=np.int32),
        policy_name=np.asarray(storage["policy_name"]),
        active_ue_count=np.asarray(storage["active_ue_count"], dtype=np.int32),
        next_active_ue_count=np.asarray(storage["next_active_ue_count"], dtype=np.int32),
        load_balance=np.asarray(storage["load_balance"], dtype=np.float32),
        blocking=np.asarray(storage["blocking"], dtype=np.float32),
        handover_count=np.asarray(storage["handover_count"], dtype=np.int32),
    )
