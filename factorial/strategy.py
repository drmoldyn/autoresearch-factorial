"""
Adaptive generation strategy for the 4-generation epoch.

After each generation, evolves the factor set:
  1. LOCK factors significant in ≥2 cumulative tests (consistent direction)
  2. DROP factors insignificant in current gen AND all prior gens (t < 0.5)
  3. EXPAND: inject Muon sub-factors when USE_MUON is significant
  4. REFINE ranges for surviving significant factors
  5. FILL freed slots with next-priority rotation factors

Gen 0 screens the full starting factor set. Gens 1-2 are adaptive.
Gen 3 (epoch end) triggers a full factor reset.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .analyzer import adaptive_locking_threshold
from .factors import (
    EPOCH_0_FACTORS,
    ROTATION_FACTORS,
    Factor,
    compute_calibration_range,
    find_factor,
    get_factor_rotation,
    get_muon_subfactors,
    get_rotation_candidates,
    is_categorical,
    make_calibrated_factor,
)

if TYPE_CHECKING:
    from .knowledge import KnowledgeStore

# Threshold below which a factor is considered dead (t-ratio)
DROP_THRESHOLD = 0.5


class GenerationStrategy:
    """Evolves the factor set between generations within an epoch."""

    def __init__(
        self,
        epoch: int,
        locked_factors: set[str],
        knowledge: KnowledgeStore,
        max_factors: int = 11,
        llm_factors: list[Factor] | None = None,
        log_fn=None,
    ):
        self.epoch = epoch
        self.locked = set(locked_factors)
        self.knowledge = knowledge
        self.max_factors = max_factors
        self.llm_factors = llm_factors
        self.log = log_fn or (lambda msg: None)

        # Accumulated within-epoch results (list per generation)
        self._gen_results: list[dict] = []
        # Factors locked during this epoch (mid-epoch locks — categoricals only)
        self._mid_epoch_locks: dict[str, float] = {}
        # Factors calibrated during this epoch (continuous factors with refined ranges)
        self._mid_epoch_calibrations: dict[str, dict] = {}
        # Current factor set (evolves each generation)
        self._current_factors: list[Factor] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_factors_for_gen(self, gen: int) -> list[Factor]:
        """Return the factor list for generation `gen`."""
        if gen == 0:
            factors = self._gen0_initial()
        else:
            factors = self._evolve_factors()

        self._current_factors = factors
        return factors

    def record_generation_result(self, gen: int, summary: dict, winner: dict):
        """Called after each generation completes."""
        self._gen_results.append({
            "gen": gen,
            "summary": summary,
            "winner": winner,
            "significant": set(summary.get("significant_factors", [])),
            "effects": summary.get("effects", {}),
            "ranked": summary.get("ranked_factors", []),
        })

    def replay_completed_gen(self, gen: int, knowledge: KnowledgeStore):
        """Replay a completed generation when resuming mid-epoch."""
        if self.epoch < len(knowledge.data.get("epochs", [])):
            epoch_data = knowledge.data["epochs"][self.epoch]
            for g_summary in epoch_data.get("generations", []):
                if g_summary.get("generation") == gen:
                    winner = g_summary.get("winner_config", {})
                    self.record_generation_result(gen, g_summary, winner)

                    # Reconstruct _current_factors from the replayed gen's
                    # factors_tested list so _evolve_factors() can work.
                    tested_names = g_summary.get("factors_tested", [])
                    if tested_names:
                        reconstructed = []
                        for name in tested_names:
                            # Check calibrating first
                            cal = knowledge.get_calibrating_factors().get(name)
                            if cal:
                                f = make_calibrated_factor(name, cal)
                                if f:
                                    reconstructed.append(f)
                                    continue
                            # Fall back to standard factor
                            f = find_factor(name)
                            if f:
                                reconstructed.append(f)
                        if reconstructed:
                            self._current_factors = reconstructed

                    # Replay any mid-epoch promotions (locks + calibrations)
                    self._check_mid_epoch_promotions()
                    return
        # If no matching generation found, create a minimal placeholder
        # so gen numbering stays correct
        self._gen_results.append({
            "gen": gen, "summary": {}, "winner": {},
            "significant": set(), "effects": {}, "ranked": [],
        })

    def get_mid_epoch_locks(self) -> dict[str, float]:
        """Return all factors locked during this epoch (for baseline update)."""
        return dict(self._mid_epoch_locks)

    def get_mid_epoch_calibrations(self) -> dict[str, dict]:
        """Return all factors calibrated during this epoch."""
        return dict(self._mid_epoch_calibrations)

    # ------------------------------------------------------------------
    # Gen 0: use epoch's starting factor set
    # ------------------------------------------------------------------

    def _gen0_initial(self) -> list[Factor]:
        """Screen the epoch's full factor set."""
        if self.llm_factors:
            factors = [f for f in self.llm_factors
                       if f.name not in self.locked][:self.max_factors]
            self.llm_factors = None
            return factors

        factors = get_factor_rotation(
            self.epoch, self.locked, self.knowledge,
        )
        return factors[:self.max_factors]

    # ------------------------------------------------------------------
    # Adaptive evolution (gen 1+)
    # ------------------------------------------------------------------

    def _evolve_factors(self) -> list[Factor]:
        """
        Core algorithm: lock → drop → expand → refine → fill.

        Returns the evolved factor set for the next generation.
        """
        if not self._gen_results or self._current_factors is None:
            # Fallback: no prior results, use gen 0 factors
            return self._gen0_initial()

        prev = self._gen_results[-1]
        winner = prev["winner"]
        ranked = prev.get("ranked", [])

        # Build lookup: factor name → (t_ratio, significant, effect)
        factor_stats: dict[str, tuple[float, bool, float]] = {}
        for entry in ranked:
            if isinstance(entry, dict):
                name = entry["name"]
                factor_stats[name] = (
                    entry.get("t_ratio", 0),
                    entry.get("significant", False),
                    entry.get("effect", 0),
                )
            elif isinstance(entry, (list, tuple)) and len(entry) >= 4:
                name, effect, t_ratio, sig = entry[0], entry[1], entry[2], entry[3]
                factor_stats[name] = (t_ratio, sig, effect)

        # ------- 1. LOCK categoricals / CALIBRATE continuous -------
        new_locks, new_cals = self._check_mid_epoch_promotions()
        for name in new_locks:
            self.log(f"  MID-EPOCH LOCK: {name} = {new_locks[name]}")
        for name, cal in new_cals.items():
            self.log(f"  MID-EPOCH CALIBRATE: {name} = {cal['best_value']} "
                     f"[{cal['range_low']:.4f}, {cal['range_high']:.4f}]")

        # ------- 2. Classify current factors -------
        keep = []       # Significant or marginal — keep testing
        drop_names = set()  # Dead weight — remove

        for f in self._current_factors:
            if f.name in self.locked:
                continue  # Already locked (including new mid-epoch locks)

            stats = factor_stats.get(f.name)
            if stats is None:
                # Factor wasn't in the design (shouldn't happen) — keep it
                keep.append(f)
                continue

            t_ratio, sig, effect = stats

            if t_ratio < DROP_THRESHOLD and not self._was_ever_significant(f.name):
                drop_names.add(f.name)
            else:
                keep.append(f)

        if drop_names:
            self.log(f"  DROPPED (t<{DROP_THRESHOLD}, never significant): "
                     f"{sorted(drop_names)}")

        # ------- 3. EXPAND: Muon sub-factors -------
        expansion = []
        if self._should_expand_muon():
            muon_subs = get_muon_subfactors()
            current_names = {f.name for f in keep}
            for mf in muon_subs:
                if (mf.name not in self.locked
                        and mf.name not in current_names
                        and mf.name not in drop_names):
                    expansion.append(mf)
                    current_names.add(mf.name)
            if expansion:
                self.log(f"  MUON EXPAND: adding {[f.name for f in expansion]}")

        # ------- 4. REFINE ranges for significant survivors -------
        # Calibrating factors use directed search; others use simple refine
        refined = []
        for f in keep:
            stats = factor_stats.get(f.name)
            if self.knowledge.is_calibrating(f.name) and stats:
                # Calibrating factor: use directed search range update
                _, _, effect = stats
                wval = winner.get(f.name, f.baseline)
                cal_low, cal_high = compute_calibration_range(
                    wval, effect, f.low, f.high, f.name,
                )
                self.knowledge.calibrate_factor(f.name, wval, cal_low, cal_high)
                refined.append(Factor(
                    name=f.name, low=cal_low, high=cal_high,
                    baseline=wval, dtype=f.dtype,
                    apply_mode=f.apply_mode, depends_on=f.depends_on,
                ))
            elif stats and stats[1] and f.dtype != "categorical":
                # Significant non-calibrating — refine around winner value
                wval = winner.get(f.name)
                if wval is not None:
                    refined.append(f.refine_around(wval))
                else:
                    refined.append(f)
            else:
                # Marginal or categorical — keep current range
                refined.append(f)

        # ------- 5. FILL freed slots -------
        all_evolved = refined + expansion
        used_names = {f.name for f in all_evolved} | self.locked
        remaining_slots = self.max_factors - len(all_evolved)

        if remaining_slots > 0:
            candidates = get_rotation_candidates(
                self.epoch, self.locked, used_names, self.knowledge,
            )
            fill = candidates[:remaining_slots]
            if fill:
                self.log(f"  FILL ({len(fill)} slots): "
                         f"{[f.name for f in fill]}")
            all_evolved.extend(fill)

        return all_evolved[:self.max_factors]

    # ------------------------------------------------------------------
    # Lock decision logic
    # ------------------------------------------------------------------

    def _check_mid_epoch_promotions(self) -> tuple[dict[str, float], dict[str, dict]]:
        """
        Check if any factors should be promoted based on cumulative evidence.

        Categorical factors → LOCK (removed from design).
        Continuous factors → CALIBRATE (stay in design with refined range).

        A factor qualifies when:
        - Significant in ≥2 cumulative tests (this epoch + historical)
        - Effect direction is consistent in SIGNIFICANT tests (≥75% same sign)

        Returns:
            (new_locks, new_calibrations) where:
            - new_locks: {name: value} for categorical factors
            - new_calibrations: {name: {best_value, range_low, range_high}}
        """
        if not self._gen_results:
            return {}, {}

        pi = self.knowledge.get_active_fraction()
        lock_t = adaptive_locking_threshold(pi)

        # Count significance and collect significant-only effects per factor
        epoch_sig_count: dict[str, int] = {}
        epoch_sig_effects: dict[str, list[float]] = {}
        for r in self._gen_results:
            for name in r["significant"]:
                epoch_sig_count[name] = epoch_sig_count.get(name, 0) + 1
                eff = r["effects"].get(name, 0)
                if name not in epoch_sig_effects:
                    epoch_sig_effects[name] = []
                epoch_sig_effects[name].append(eff)

        new_locks = {}
        new_calibrations = {}
        latest_winner = self._gen_results[-1]["winner"]

        for name, epoch_count in epoch_sig_count.items():
            if name in self.locked or name in self._mid_epoch_locks:
                continue
            if name in self._mid_epoch_calibrations:
                continue  # Already calibrating — range update happens in _evolve

            # Use knowledge store's factor_history as the single source of truth.
            # factor_history is already updated via knowledge.record_generation()
            # after each gen, so it includes current epoch's results.
            # Using epoch_count + hist_sig would DOUBLE-COUNT this epoch.
            hist = self.knowledge.data.get("factor_history", {}).get(name, {})
            total_sig = hist.get("significant_count", 0)
            if total_sig < 2:
                continue  # Need ≥2 significant tests

            # Direction consistency check (≥75% of significant effects same sign)
            hist_sig_effects = hist.get("significant_effect_sizes", [])
            if not hist_sig_effects:
                hist_sig_effects = hist.get("effect_sizes", [])
            sig_effects = list(hist_sig_effects)  # Already includes epoch effects
            nonzero = [e for e in sig_effects if e != 0]
            if not nonzero:
                continue
            n_pos = sum(1 for e in nonzero if e > 0)
            n_neg = len(nonzero) - n_pos
            dominant_count = max(n_pos, n_neg)
            if dominant_count / len(nonzero) < 0.75:
                continue  # Direction too inconsistent

            best_val = latest_winner.get(name)
            if best_val is None:
                continue

            if is_categorical(name):
                # Categorical → LOCK
                new_locks[name] = best_val
                self._mid_epoch_locks[name] = best_val
                self.locked.add(name)
                self.knowledge.lock_factors({name: best_val})
            else:
                # Continuous → CALIBRATE with directed range
                latest_effect = epoch_sig_effects.get(name, [0])[-1]
                # Find tested range from current factors
                tested_low, tested_high = best_val, best_val
                if self._current_factors:
                    for f in self._current_factors:
                        if f.name == name:
                            tested_low = f.low
                            tested_high = f.high
                            break
                cal_low, cal_high = compute_calibration_range(
                    best_val, latest_effect, tested_low, tested_high, name,
                )
                cal_data = {
                    "best_value": best_val,
                    "range_low": cal_low,
                    "range_high": cal_high,
                }
                new_calibrations[name] = cal_data
                self._mid_epoch_calibrations[name] = cal_data
                self.knowledge.calibrate_factor(name, best_val, cal_low, cal_high)

        return new_locks, new_calibrations

    def _was_ever_significant(self, name: str) -> bool:
        """Check if a factor was significant in any generation this epoch."""
        for r in self._gen_results:
            if name in r["significant"]:
                return True
        return False

    def _should_expand_muon(self) -> bool:
        """Check if USE_MUON is significant and sub-factors aren't already present."""
        # Was USE_MUON significant in any generation this epoch?
        muon_significant = False
        for r in self._gen_results:
            if "USE_MUON" in r["significant"]:
                muon_significant = True
                break

        if not muon_significant:
            return False

        # Are Muon sub-factors already in the current factor set?
        if self._current_factors:
            current_names = {f.name for f in self._current_factors}
            muon_sub_names = {f.name for f in get_muon_subfactors()}
            if muon_sub_names & current_names:
                return False  # Already present

        return True
