import numpy as np


class FactoredMLBMixin:
    DEFAULT_OBSERVATION_COMPONENTS = (
        "rsrp",
        "cio",
        "association",
    )

    def _runtime_devices(self):
        """Return the devices that participate in the current transition."""
        return self.active_devices

    def _observation_slots(self):
        """Return fixed observation slots; ``None`` denotes an inactive slot."""
        return self.device_slots

    def _observation_capacity(self):
        return len(self._observation_slots())

    def _radio_value(self, key, default):
        radio_config = self.config.get("radio_config", {})
        if key in radio_config:
            return radio_config[key]

        network_config = self.config.get("network_config", {})
        aliases = {
            "prb_bandwidth": "prb_bandwidth_hz",
            "user_demand_bps": "user_demand",
        }
        if key in network_config:
            return network_config[key]
        alias = aliases.get(key)
        if alias in network_config:
            return network_config[alias]
        return default

    def _total_prbs(self):
        return int(self._radio_value("total_prbs", 50))

    def _prb_bandwidth(self):
        return float(self._radio_value("prb_bandwidth", 180000.0))

    def _user_demand_bps(self):
        return float(self._radio_value("user_demand_bps", self.config.get("user_demand", 1024 * 1000)))

    def _max_throughput_per_bs(self):
        return self._total_prbs() * 5.0 * self._prb_bandwidth()

    def _configure_factored_spaces(self):
        self.num_base_stations = len(self.base_stations)
        self.num_devices = self._observation_capacity()
        self.rsrp_dim = self.num_base_stations * self.num_devices
        self.action_dim = self.num_base_stations * (self.num_base_stations - 1)

        obs_config = self.config.get("observation_config", {})
        obs_type = obs_config.get("type", "factored")
        if obs_type == "rsrp":
            self.observation_components = ("rsrp",)
        else:
            self.observation_components = tuple(
                obs_config.get("components", self.DEFAULT_OBSERVATION_COMPONENTS)
            )

        self.observation_slices = {}
        start = 0
        for component in self.observation_components:
            dim = self._observation_component_dim(component)
            self.observation_slices[component] = slice(start, start + dim)
            start += dim

        self.state_dim = start

    def _observation_component_dim(self, component):
        if component == "rsrp":
            return self.rsrp_dim
        if component == "cio":
            return self.action_dim
        if component == "association":
            return self.num_base_stations * self.num_devices
        if component == "measurement_mask":
            return self.num_base_stations * self.num_devices
        if component == "load":
            return self.num_base_stations
        if component == "handover":
            return self.num_devices
        if component in ("active_mask", "connected_mask"):
            return self.num_devices
        raise ValueError(f"Unknown observation component: {component}")

    def _reset_transition_metrics(self):
        self.last_handover_count = 0
        self.last_pingpong_count = 0
        self.last_handover_baseline = 0
        self.last_pingpong_baseline = 0
        self.last_cio_change_magnitude = 0.0
        for device in self._runtime_devices():
            device["recent_handover"] = 0.0
            device["previous_connected_bs_id"] = None

    def _get_hypothetical_connections(self, cio_matrix, return_connections=False):
        """Evaluate a CIO configuration using the current, unchanged references."""
        handover_count = 0
        pingpong_count = 0
        connections = {} if return_connections else None
        
        hysteresis_enabled = self.config.get("hysteresis_config", {}).get("enabled", True)
        hysteresis_value = self.config.get("hysteresis_config", {}).get("value", 2.0)
        
        for device in self._runtime_devices():
            if not device.get("connected_allowed", True):
                continue
            best_bs = None
            best_score = -np.inf
            current_bs = device["connected_bs"]
            
            for bs in self.base_stations:
                bs_id = int(bs["id"])
                if bs_id not in device["rsrp_measurements"]:
                    continue
                rsrp = device["rsrp_measurements"][bs_id]
                
                if current_bs is not None and current_bs["id"] != bs_id:
                    cio = (
                        cio_matrix.get((current_bs["id"], bs_id), 0.0)
                        - cio_matrix.get((bs_id, current_bs["id"]), 0.0)
                    )
                else:
                    cio = 0
                
                score = rsrp + cio
                
                if hysteresis_enabled and current_bs is not None and bs_id == current_bs["id"]:
                    score += hysteresis_value
                
                if score > best_score:
                    best_score = score
                    best_bs = bs
            
            current_bs_id = current_bs["id"] if current_bs is not None else None
            best_bs_id = best_bs["id"] if best_bs is not None else None
            if return_connections:
                connections[int(device["id"])] = best_bs_id
            
            if current_bs_id != best_bs_id:
                handover_count += 1
                if device.get("previous_connected_bs_id") == best_bs_id:
                    pingpong_count += 1
                    
        return connections if return_connections else (handover_count, pingpong_count)

    def _cio_bounds(self):
        cio_config = self.config.get("cio_config", {})
        min_value = cio_config.get("min_value", -6)
        max_value = cio_config.get("max_value", 6)
        return min_value, max_value

    def _cio_abs_bound(self):
        min_value, max_value = self._cio_bounds()
        return max(abs(min_value), abs(max_value), 1.0)

    def _apply_cio_action(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.action_dim:
            raise ValueError(
                f"Expected action_dim={self.action_dim}, got {action.shape[0]}"
            )
        action_config = self.config.get("action_config", {})
        action_mode = action_config.get("mode", "delta")
        delta_scale = action_config.get(
            "delta_scale", self.config.get("cio_config", {}).get("step_size", 0.5)
        )
        min_cio, max_cio = self._cio_bounds()
        cio_abs_bound = self._cio_abs_bound()

        changes = []
        action_index = 0
        for i, bs_i in enumerate(self.base_stations):
            for j, bs_j in enumerate(self.base_stations):
                if i == j:
                    continue

                key = (bs_i["id"], bs_j["id"])
                old_cio = self.cio_matrix.get(key, 0.0)
                if action_mode == "delta":
                    raw_cio = old_cio + action[action_index] * delta_scale
                else:
                    raw_cio = action[action_index] * cio_abs_bound

                new_cio = max(min_cio, min(max_cio, raw_cio))
                self.cio_matrix[key] = new_cio
                changes.append(abs(new_cio - old_cio) / cio_abs_bound)
                action_index += 1

        self.last_cio_change_magnitude = float(np.mean(changes)) if changes else 0.0

    def _get_rsrp_observation(self):
        state = []
        inactive_value = float(
            self.config.get("observation_config", {}).get("inactive_rsrp", -120.0)
        )
        for bs in self.base_stations:
            for device in self._observation_slots():
                if device is None:
                    state.append(inactive_value)
                else:
                    state.append(
                        device["rsrp_measurements"].get(bs["id"], inactive_value)
                    )
        return np.array(state, dtype=np.float32)

    def _get_cio_observation(self):
        values = []
        min_cio, max_cio = self._cio_bounds()
        cio_range = max(float(max_cio - min_cio), 1e-8)
        for i, bs_i in enumerate(self.base_stations):
            for j, bs_j in enumerate(self.base_stations):
                if i != j:
                    values.append(
                        (
                            self.cio_matrix.get((bs_i["id"], bs_j["id"]), 0.0)
                            - min_cio
                        )
                        / cio_range
                    )
        return np.array(values, dtype=np.float32)

    def _get_association_observation(self):
        values = []
        for bs in self.base_stations:
            for device in self._observation_slots():
                if device is None:
                    values.append(0.0)
                    continue
                connected_bs = device.get("connected_bs")
                values.append(
                    1.0
                    if connected_bs is not None and connected_bs["id"] == bs["id"]
                    else 0.0
                )
        return np.array(values, dtype=np.float32)

    def _get_measurement_mask_observation(self):
        values = []
        for bs in self.base_stations:
            bs_id = int(bs["id"])
            for device in self._observation_slots():
                if device is None:
                    values.append(0.0)
                    continue
                measurement_mask = device.get("measurement_mask")
                if measurement_mask is None:
                    measured = bs_id in device.get("rsrp_measurements", {})
                else:
                    measured = bool(measurement_mask.get(bs_id, False))
                values.append(1.0 if measured else 0.0)
        return np.array(values, dtype=np.float32)

    def _get_load_observation(self):
        return np.array(
            [bs.get("resource_utilization", 0.0) for bs in self.base_stations],
            dtype=np.float32,
        )

    def _get_handover_observation(self):
        return np.array(
            [
                device.get("recent_handover", 0.0) if device is not None else 0.0
                for device in self._observation_slots()
            ],
            dtype=np.float32,
        )

    def _get_active_mask_observation(self):
        return np.array(
            [1.0 if device is not None else 0.0 for device in self._observation_slots()],
            dtype=np.float32,
        )

    def _get_connected_mask_observation(self):
        return np.array(
            [
                1.0
                if (
                    device is not None
                    and device.get("connected_allowed", True)
                    and device.get("connected_bs") is not None
                )
                else 0.0
                for device in self._observation_slots()
            ],
            dtype=np.float32,
        )

    def _build_observation(self):
        component_values = {
            "rsrp": self._get_rsrp_observation,
            "cio": self._get_cio_observation,
            "association": self._get_association_observation,
            "measurement_mask": self._get_measurement_mask_observation,
            "load": self._get_load_observation,
            "handover": self._get_handover_observation,
            "active_mask": self._get_active_mask_observation,
            "connected_mask": self._get_connected_mask_observation,
        }
        parts = [component_values[name]() for name in self.observation_components]
        return np.concatenate(parts).astype(np.float32)

    def _split_observation(self, observation):
        observation = np.asarray(observation, dtype=np.float32).reshape(-1)
        if observation.shape[0] != self.state_dim:
            raise ValueError(
                f"Expected observation dim {self.state_dim}, got {observation.shape[0]}"
            )
        return {
            component: observation[obs_slice].copy()
            for component, obs_slice in self.observation_slices.items()
        }

    def _get_rsrp_matrix(self):
        return self._get_rsrp_observation().reshape(
            self.num_base_stations, self.num_devices
        )

    def _get_serving_cell_ids(self):
        serving_ids = []
        for device in self._observation_slots():
            if device is None:
                serving_ids.append(-1)
                continue
            connected_bs = device.get("connected_bs")
            serving_ids.append(connected_bs["id"] if connected_bs is not None else -1)
        return np.array(serving_ids, dtype=np.int32)

    def _structured_mlb_reward(self, throughput, blocking, use_cf_debias=False):
        reward_config = self.config.get("reward_config", {})
        loads = [bs.get("resource_utilization", 0.0) for bs in self.base_stations]
        load_std = float(np.std(loads)) if loads else 0.0

        if use_cf_debias:
            ho_count = max(0, self.last_handover_count - getattr(self, "last_handover_baseline", 0))
        else:
            ho_count = self.last_handover_count

        active_device_count = len(self._runtime_devices())
        handover_rate = ho_count / active_device_count if active_device_count else 0.0

        return (
            -reward_config.get("load_weight", 1.0) * load_std
            - reward_config.get("handover_weight", 1.0) * handover_rate
        )
