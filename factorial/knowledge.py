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
                return json.loads(content)
        return {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "global_best_val_bpb": float("inf"),
            "global_best_config": {},
            "locked_factors": {},
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
        Return confidence level for a factor: "high", "medium", "low", "untested".
        Based on number of tests and consistency of significance.
        """
        if name in self.data.get("locked_factors", {}):
            return "locked"

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
        Suggest factors that should be locked based on accumulated evidence.

        A factor is a lock candidate if:
        - Significant >= 2 times
        - Effect direction is consistent in ≥75% of significant tests
          (insignificant runs produce near-zero noise with random sign)

        Args:
            locking_threshold: Not used for direct t-ratio comparison here
                (we don't store per-test t-ratios), but the criteria below
                approximate the same bar through repeated significance.
        """
        candidates = []
        for name, hist in self.data.get("factor_history", {}).items():
            if name in self.data.get("locked_factors", {}):
                continue
            if hist["significant_count"] >= 2:
                # Use significant-only effects if available, else all effects
                effects = hist.get("significant_effect_sizes",
                                   hist.get("effect_sizes", []))
                nonzero = [e for e in effects if e != 0]
                if not nonzero:
                    continue
                # Supermajority direction check (≥75% same sign)
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
        Auto-lock factors that are high-confidence but can never pass the
        direction consistency check for locking. These factors have been
        tested many times (significant enough to qualify) but their effect
        direction is too noisy to establish a clear best level.

        Rather than wasting design slots testing them forever, graduate
        them to locked at their baseline value. The baseline is already
        the best-performing config, so this is safe.

        Returns dict of graduated factor names -> locked values.
        """
        log = log_fn or (lambda msg: None)
        graduated = {}
        for name, hist in self.data.get("factor_history", {}).items():
            if name in self.data.get("locked_factors", {}):
                continue
            total = hist.get("total_tests", 0)
            sig = hist.get("significant_count", 0)
            if total >= min_tests and sig >= min_sig:
                # This factor has been tested extensively and is significant,
                # but hasn't been locked (likely due to direction inconsistency).
                # Graduate it to locked at baseline.
                lock_val = baseline.get(name)
                if lock_val is not None:
                    graduated[name] = lock_val
                    log(f"  GRADUATED (tested {total}x, sig {sig}x, "
                        f"direction inconsistent): {name} = {lock_val}")

        if graduated:
            self.lock_factors(graduated)
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
