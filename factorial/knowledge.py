"""
Persistent knowledge store for factorial screening.

Tracks all accumulated knowledge across epochs and generations:
- Factor effect sizes and significance
- Locked factors (high-confidence, don't re-test)
- Best configurations at each checkpoint
- Factor interaction hints from foldover results

JSON-backed for crash recovery and human inspection.
"""

import json
import time
from pathlib import Path


class KnowledgeStore:
    """
    JSON-backed store of everything learned across all epochs.
    Automatically persists after every mutation.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            content = self.path.read_text().strip()
            if content:
                data = json.loads(content)
                # Schema migration: add calibrating_factors if missing
                if "calibrating_factors" not in data:
                    data["calibrating_factors"] = {}
                return data
        return {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "global_best_val_bpb": float("inf"),
            "global_best_config": {},
            "locked_factors": {},
            "calibrating_factors": {},
            "factor_history": {},
            "epochs": [],
        }

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, default=str))

    def record_generation(
        self,
        epoch: int,
        gen: int,
        summary: dict,
    ):
        """Record a generation's results."""
        # Ensure epoch entry exists
        while len(self.data["epochs"]) <= epoch:
            self.data["epochs"].append({
                "epoch": len(self.data["epochs"]),
                "generations": [],
                "epoch_best_val_bpb": float("inf"),
                "epoch_best_config": {},
            })

        epoch_data = self.data["epochs"][epoch]
        epoch_data["generations"].append(summary)

        # Update epoch best
        best_bpb = summary.get("best_val_bpb")
        if best_bpb is not None and best_bpb < epoch_data["epoch_best_val_bpb"]:
            epoch_data["epoch_best_val_bpb"] = best_bpb
            epoch_data["epoch_best_config"] = summary.get("winner_config", {})

        # Update global best
        if best_bpb is not None and best_bpb < self.data["global_best_val_bpb"]:
            self.data["global_best_val_bpb"] = best_bpb
            self.data["global_best_config"].update(summary.get("winner_config", {}))

        # Update factor history
        sig_set = set(summary.get("significant_factors", []))
        for name in summary.get("factors_tested", []):
            if name not in self.data["factor_history"]:
                self.data["factor_history"][name] = {
                    "tested_epochs": [],
                    "effect_sizes": [],
                    "significant_effect_sizes": [],
                    "significant_count": 0,
                    "total_tests": 0,
                }
            hist = self.data["factor_history"][name]
            # Ensure significant_effect_sizes exists (upgrade old format)
            if "significant_effect_sizes" not in hist:
                hist["significant_effect_sizes"] = []
            if epoch not in hist["tested_epochs"]:
                hist["tested_epochs"].append(epoch)
            effect = summary.get("effects", {}).get(name, 0)
            hist["effect_sizes"].append(effect)
            hist["total_tests"] += 1
            if name in sig_set:
                hist["significant_count"] += 1
                hist["significant_effect_sizes"].append(effect)

        self._save()

    def lock_factors(self, factors: dict[str, float]):
        """Lock factors at their best known values (high confidence)."""
        self.data["locked_factors"].update(
            {k: v for k, v in factors.items()}
        )
        self._save()

    def get_locked_factors(self) -> dict[str, float]:
        """Return all locked factor values."""
        return dict(self.data.get("locked_factors", {}))

    def unlock_factors(self, names: set[str]):
        """Remove locks from specified factors (e.g., Muon dependency clearing)."""
        for name in names:
            self.data.get("locked_factors", {}).pop(name, None)
        self._save()

    # ------------------------------------------------------------------
    # Calibration tier: continuous factors with progressively refined ranges
    # ------------------------------------------------------------------

    def calibrate_factor(
        self,
        name: str,
        best_value: float,
        range_low: float,
        range_high: float,
    ):
        """Add or update a factor in calibration mode.

        Calibrating factors stay in the PB design with refined ranges
        (directed search). Unlike locked factors, they are never removed
        from the design — allowing detection of interaction effects.
        """
        cal = self.data.setdefault("calibrating_factors", {})
        if name in cal:
            cal[name]["best_value"] = best_value
            cal[name]["range_low"] = range_low
            cal[name]["range_high"] = range_high
            cal[name]["n_calibrations"] = cal[name].get("n_calibrations", 0) + 1
        else:
            cal[name] = {
                "best_value": best_value,
                "range_low": range_low,
                "range_high": range_high,
                "epoch_entered": len(self.data.get("epochs", [])),
                "n_calibrations": 0,
            }
        # Ensure the factor is NOT also locked (calibration replaces locking)
        self.data.get("locked_factors", {}).pop(name, None)
        self._save()

    def get_calibrating_factors(self) -> dict:
        """Return all calibrating factors: {name: {best_value, range_low, range_high, ...}}."""
        return dict(self.data.get("calibrating_factors", {}))

    def is_calibrating(self, name: str) -> bool:
        """Check if a factor is in calibration mode."""
        return name in self.data.get("calibrating_factors", {})

    def uncalibrate_factor(self, name: str):
        """Remove a factor from calibration (e.g., if it becomes consistently insignificant)."""
        self.data.get("calibrating_factors", {}).pop(name, None)
        self._save()

    def get_best_config(self) -> dict:
        """Return the global best configuration."""
        return dict(self.data.get("global_best_config", {}))

    def get_best_val_bpb(self) -> float:
        """Return the global best val_bpb."""
        return self.data.get("global_best_val_bpb", float("inf"))

    def get_epoch_best(self, epoch: int) -> dict:
        """Return the best config and val_bpb for a specific epoch."""
        if epoch < len(self.data["epochs"]):
            ep = self.data["epochs"][epoch]
            return {
                "config": ep.get("epoch_best_config", {}),
                "val_bpb": ep.get("epoch_best_val_bpb", float("inf")),
            }
        return {"config": {}, "val_bpb": float("inf")}

    def record_validation(self, epoch: int, val_bpb: float, config: dict):
        """Record a post-epoch validation result (solo run of evolved baseline).

        Unlike PB screening runs, validation runs use the full evolved baseline
        with no factor perturbation. They measure true accumulated improvement.
        """
        if "validations" not in self.data:
            self.data["validations"] = []

        self.data["validations"].append({
            "epoch": epoch,
            "val_bpb": val_bpb,
            "config": config,
            "timestamp": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        })

        # Update global best if this validation beats it
        if val_bpb < self.data.get("global_best_val_bpb", float("inf")):
            self.data["global_best_val_bpb"] = val_bpb
            self.data["global_best_config"] = dict(config)

        self._save()

    def get_factor_confidence(self, name: str) -> str:
        """
        Return confidence level for a factor:
        "locked", "calibrating", "high", "medium", "low", "untested".

        Locked: binary/categorical factors permanently set.
        Calibrating: continuous factors with proven significance undergoing
                     directed range refinement (never removed from design).
        """
        if name in self.data.get("locked_factors", {}):
            return "locked"

        if name in self.data.get("calibrating_factors", {}):
            return "calibrating"

        hist = self.data.get("factor_history", {}).get(name)
        if hist is None:
            return "untested"

        total = hist["total_tests"]
        sig = hist["significant_count"]

        if total >= 3 and sig >= 2:
            return "high"
        elif total >= 2 and sig >= 1:
            return "medium"
        elif total >= 1:
            return "low"
        return "untested"

    def get_latest_effect(self, name: str) -> float | None:
        """Return the most recent effect size for a factor."""
        hist = self.data.get("factor_history", {}).get(name)
        if hist and hist["effect_sizes"]:
            return hist["effect_sizes"][-1]
        return None

    def suggest_lock_candidates(self, locking_threshold: float = 2.5) -> list[tuple[str, float]]:
        """
        Suggest CATEGORICAL factors that should be locked based on accumulated evidence.

        Only binary/categorical factors are eligible for locking. Continuous
        factors are sent to calibration instead (see suggest_calibration_candidates).

        A factor is a lock candidate if:
        - It is categorical (binary toggle, code-swap)
        - Significant >= 2 times
        - Effect direction is consistent in ≥75% of significant tests
        """
        from .factors import is_categorical

        candidates = []
        for name, hist in self.data.get("factor_history", {}).items():
            if name in self.data.get("locked_factors", {}):
                continue
            if name in self.data.get("calibrating_factors", {}):
                continue
            if not is_categorical(name):
                continue  # Continuous → calibrate, don't lock
            if hist["significant_count"] >= 2:
                effects = hist.get("significant_effect_sizes",
                                   hist.get("effect_sizes", []))
                nonzero = [e for e in effects if e != 0]
                if not nonzero:
                    continue
                n_pos = sum(1 for e in nonzero if e > 0)
                n_neg = len(nonzero) - n_pos
                dominant = max(n_pos, n_neg)
                if dominant / len(nonzero) >= 0.75:
                    avg_effect = sum(effects) / len(effects)
                    candidates.append((name, avg_effect))

        candidates.sort(key=lambda x: abs(x[1]), reverse=True)
        return candidates

    def suggest_calibration_candidates(self) -> list[tuple[str, float]]:
        """
        Suggest CONTINUOUS factors that should enter calibration mode.

        A factor enters calibration when:
        - It is NOT categorical
        - It is not already locked or calibrating
        - Significant >= 2 times
        - Effect direction is consistent in ≥75% of significant tests

        Returns list of (name, avg_effect) sorted by |effect| descending.
        """
        from .factors import is_categorical

        candidates = []
        for name, hist in self.data.get("factor_history", {}).items():
            if name in self.data.get("locked_factors", {}):
                continue
            if name in self.data.get("calibrating_factors", {}):
                continue
            if is_categorical(name):
                continue  # Categoricals → lock, don't calibrate
            if hist["significant_count"] >= 2:
                effects = hist.get("significant_effect_sizes",
                                   hist.get("effect_sizes", []))
                nonzero = [e for e in effects if e != 0]
                if not nonzero:
                    continue
                n_pos = sum(1 for e in nonzero if e > 0)
                n_neg = len(nonzero) - n_pos
                dominant = max(n_pos, n_neg)
                if dominant / len(nonzero) >= 0.75:
                    avg_effect = sum(effects) / len(effects)
                    candidates.append((name, avg_effect))

        candidates.sort(key=lambda x: abs(x[1]), reverse=True)
        return candidates

    def graduate_stale_factors(
        self,
        baseline: dict[str, float],
        min_tests: int = 10,
        min_sig: int = 5,
        log_fn=None,
    ) -> dict[str, float]:
        """
        Graduate factors that have been tested extensively but can't pass
        the direction consistency check for locking/calibration.

        - Categorical factors: graduate to LOCKED at baseline value.
        - Continuous factors: graduate to CALIBRATING with narrow range
          around baseline (never fully locked).

        Returns dict of graduated factor names -> values (locked or calibrated).
        """
        from .factors import compute_calibration_range, is_categorical

        log = log_fn or (lambda msg: None)
        graduated = {}
        for name, hist in self.data.get("factor_history", {}).items():
            if name in self.data.get("locked_factors", {}):
                continue
            if name in self.data.get("calibrating_factors", {}):
                continue
            total = hist.get("total_tests", 0)
            sig = hist.get("significant_count", 0)
            if total >= min_tests and sig >= min_sig:
                grad_val = baseline.get(name)
                if grad_val is None:
                    continue

                if is_categorical(name):
                    # Categorical: lock at baseline
                    graduated[name] = grad_val
                    log(f"  GRADUATED→LOCKED (tested {total}x, sig {sig}x, "
                        f"direction inconsistent): {name} = {grad_val}")
                else:
                    # Continuous: calibrate with narrow range around baseline
                    # Use a small range (±10% of the last tested span, or
                    # the average effect size as a guide)
                    effects = hist.get("significant_effect_sizes",
                                       hist.get("effect_sizes", []))
                    avg_eff = sum(effects) / len(effects) if effects else 0
                    from .factors import get_factor_bounds
                    orig_low, orig_high = get_factor_bounds(name)
                    narrow_span = (orig_high - orig_low) * 0.1
                    cal_low = max(orig_low, grad_val - narrow_span)
                    cal_high = min(orig_high, grad_val + narrow_span)
                    self.calibrate_factor(name, grad_val, cal_low, cal_high)
                    graduated[name] = grad_val
                    log(f"  GRADUATED→CALIBRATING (tested {total}x, sig {sig}x, "
                        f"direction inconsistent): {name} [{cal_low:.4f}, {cal_high:.4f}]")

        # Lock only the categoricals
        categoricals_to_lock = {
            n: v for n, v in graduated.items() if is_categorical(n)
        }
        if categoricals_to_lock:
            self.lock_factors(categoricals_to_lock)
        return graduated

    def get_active_fraction(self, lookback: int = 5, learning_rate: float = 0.2) -> float:
        """
        Estimate π — the fraction of factors that are active — from recent data.

        Uses an exponential moving average anchored to the prior (π₀ = 0.25)
        with a low learning rate to prevent overfitting to noisy early data.

        π_adapted = (1 - α) * π₀ + α * π_observed

        where α = learning_rate * min(1, n_generations / lookback).
        The effective α ramps from 0 → learning_rate as data accumulates,
        so we never move aggressively from the prior on sparse data.

        Returns π clamped to [0.05, 0.50]. Returns 0.25 (the prior) if no
        data is available yet.
        """
        PI_PRIOR = 0.25

        recent_gens = []
        for ep in reversed(self.data.get("epochs", [])):
            for gen in reversed(ep.get("generations", [])):
                recent_gens.append(gen)
                if len(recent_gens) >= lookback:
                    break
            if len(recent_gens) >= lookback:
                break

        if not recent_gens:
            return PI_PRIOR

        total_tested = 0
        total_significant = 0
        for gen in recent_gens:
            n_tested = len(gen.get("factors_tested", []))
            n_sig = len(gen.get("significant_factors", []))
            total_tested += n_tested
            total_significant += n_sig

        if total_tested == 0:
            return PI_PRIOR

        pi_observed = total_significant / total_tested

        # Ramp effective learning rate based on data volume
        n_gens = len(recent_gens)
        effective_alpha = learning_rate * min(1.0, n_gens / lookback)

        # EMA toward observed, anchored at prior
        pi_adapted = (1 - effective_alpha) * PI_PRIOR + effective_alpha * pi_observed
        return max(0.05, min(0.50, pi_adapted))

    @property
    def total_experiments(self) -> int:
        """Total number of experiments run across all epochs."""
        total = 0
        for ep in self.data.get("epochs", []):
            for gen in ep.get("generations", []):
                total += gen.get("n_experiments", 0)
        return total
