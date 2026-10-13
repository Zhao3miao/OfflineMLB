import copy

from .cio_load_balancing_env import CIOLoadBalancingEnv
from .mro_snapshot import MROSnapshotScenario


def _deep_update(target, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


class RealDataCIOLoadBalancingEnv(CIOLoadBalancingEnv):
    """CIO environment driven by the supplied anonymized RSRP snapshots."""

    def __init__(self, processed_dir, config_overrides=None):
        scenario = MROSnapshotScenario(processed_dir)
        config = copy.deepcopy(scenario.config)
        if config_overrides:
            _deep_update(config, config_overrides)
        super().__init__(config, scenario=scenario)

    def _measure_rsrp(self):
        # _refresh_active_devices copies measurements from the current snapshot.
        return None

    def _build_info(self, *args, **kwargs):
        info = super()._build_info(*args, **kwargs)
        if self.current_time < self.total_time:
            info["timestamp"] = str(self.scenario.timestamps[self.current_time])
        else:
            info["timestamp"] = None
        return info
