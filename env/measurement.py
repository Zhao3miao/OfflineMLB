"""Action-independent reporting for simulated RSRP (never used for MRO replay)."""

import hashlib
import json

import numpy as np


class MeasurementReporter:
    def __init__(self, config, cell_ids, scenario_key=""):
        self.cell_ids = tuple(sorted(int(value) for value in cell_ids))
        self.config = dict(config or {})
        self.mode = self.config.get("mode", "full")
        if self.mode not in ("full", "sparse"):
            raise ValueError("measurement mode must be full or sparse")
        if self.config.get("version", 1) != 1:
            raise ValueError("Unsupported measurement version")
        self.seed = int(self.config.get("seed", 0))
        self.scenario_key = str(scenario_key)
        self.probs = np.asarray(
            self.config.get("report_count_probs", [1 / len(self.cell_ids)] * len(self.cell_ids)),
            dtype=float,
        )
        if (self.probs.shape != (len(self.cell_ids),) or not np.all(np.isfinite(self.probs))
                or np.any(self.probs < 0) or not np.isclose(self.probs.sum(), 1.0)):
            raise ValueError("report_count_probs must be finite, nonnegative and sum to one, one entry per cell")
        self.threshold = self.config.get("neighbor_rsrp_threshold_dbm")
        if self.threshold is not None:
            self.threshold = float(self.threshold)
            if not np.isfinite(self.threshold):
                raise ValueError("neighbor_rsrp_threshold_dbm must be finite")
        self._budgets = {}

    def budget(self, ue_id):
        """One reproducible report budget per exogenous UE lifecycle."""
        if ue_id not in self._budgets:
            key = json.dumps([self.seed, self.scenario_key, int(ue_id)], separators=(",", ":"))
            seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
            rng = np.random.default_rng(seed)
            self._budgets[ue_id] = int(rng.choice(np.arange(1, len(self.cell_ids) + 1), p=self.probs))
        return self._budgets[ue_id]

    def report(self, full_rsrp, ue_id):
        if set(full_rsrp) != set(self.cell_ids) or not all(np.isfinite(v) for v in full_rsrp.values()):
            raise ValueError("Complete finite RSRP is required before reporting")
        ranked = sorted(self.cell_ids, key=lambda cell: (-full_rsrp[cell], cell))
        selected = ranked
        if self.mode == "sparse":
            neighbors = [cell for cell in ranked[1:]
                         if self.threshold is None or full_rsrp[cell] >= self.threshold]
            selected = ranked[:1] + neighbors[:self.budget(ue_id) - 1]
        measured = {cell: float(full_rsrp[cell]) for cell in selected}
        mask = {cell: cell in measured for cell in self.cell_ids}
        return measured, mask
