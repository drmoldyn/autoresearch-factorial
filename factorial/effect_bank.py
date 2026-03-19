"""
Pooled cross-generation meta-analysis of main effects and 2-factor interactions.

Accumulates evidence from every PB screening generation. Provides:
- Precision-weighted pooled main effect estimates
- Within-generation interaction contrasts (A×B from quadrant partitioning)
- Cross-generation pooled interaction estimates
- Hypothesis-driven config construction that accounts for interactions

The key insight: in PB(12,11), each factor pair (A,B) has 3 runs per quadrant
(A+B+, A+B-, A-B+, A-B-). The interaction contrast
    I_AB = (mean(A+B+ ∪ A-B-) - mean(A+B- ∪ A-B+)) / 2
is unbiased even under PB aliasing, because we condition on observed levels.

Pooling across generations with different factor rotations provides additional
de-aliasing and greatly increases statistical power.
"""

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .designer import generate_pb_design
from .factors import ALL_FACTORS, find_factor

# Minimum SE floor to prevent near-zero Lenth's PSE from dominating
# the precision-weighted pool. With PB(12,11) and val_bpb in [1.3, 1.7],
# typical SE is 0.02-0.05. Floor at 0.005 prevents degenerate weighting.
MIN_SE = 0.005

# Maximum effect magnitude. Effects beyond this are Winsorized (clamped).
# In PB(12,11), the effect distribution has p99=0.27 with a gap to 0.54.
# Real large effects (e.g. UNEMBEDDING_LR threshold at 0.56) exist in
# the 0.4-0.6 range. Cap at 0.6 preserves these while protecting against
# inf-driven artifacts (which produce effects >1.0).
MAX_EFFECT = 0.6

# Crash-rate SE inflation. When a generation has >25% crashed runs, the
# surviving runs are a biased sample (crashes correlate with factor levels).
# Inflate SE by this factor to down-weight crash-contaminated generations
# without excluding them entirely. This targets the root cause (crash bias)
# rather than a proxy (large effect magnitude).
CRASH_SE_INFLATION = 2.5
CRASH_RATE_THRESHOLD = 0.25


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GenerationRecord:
    """One generation's worth of data for the effect bank."""
    epoch: int
    generation: int
    arm: str
    factors_tested: list[str]
    effects: dict[str, float]       # main effects from PB contrast
    se: float                        # Lenth's PSE
    responses: list[float]           # per-run val_bpb (12 values for PB12)
    n_runs: int
    n_crashes: int = 0               # runs that diverged/crashed to inf
    interaction_contrasts: dict[str, float] = field(default_factory=dict)
    interaction_se: float = 0.0

    @property
    def crash_rate(self) -> float:
        return self.n_crashes / self.n_runs if self.n_runs > 0 else 0.0

    @property
    def effective_se(self) -> float:
        """SE with crash-rate inflation applied."""
        se = max(self.se, MIN_SE)
        if self.crash_rate > CRASH_RATE_THRESHOLD:
            se *= CRASH_SE_INFLATION
        return se


@dataclass
class PooledMainEffect:
    """Precision-weighted pooled estimate of a factor's main effect."""
    factor: str
    n_generations: int
    weighted_effect: float      # precision-weighted mean
    pooled_se: float            # 1/sqrt(sum(1/se_i^2))
    pooled_t: float             # |weighted_effect| / pooled_se
    direction_consistency: float  # fraction of gens with same sign as pooled
    individual_effects: list[float] = field(default_factory=list)


@dataclass
class PooledInteraction:
    """Precision-weighted pooled estimate of a 2-factor interaction."""
    factor_a: str
    factor_b: str
    n_generations: int
    weighted_contrast: float
    pooled_se: float
    pooled_t: float
    # Conditional effects: optimal direction of A given B's level
    conditional_a_given_b_plus: float = 0.0   # effect of A when B=+1
    conditional_a_given_b_minus: float = 0.0  # effect of A when B=-1


# ---------------------------------------------------------------------------
# EffectBank
# ---------------------------------------------------------------------------

class EffectBank:
    """
    Pooled meta-analysis of main effects and 2-factor interactions
    across all PB screening generations.

    Used at epoch boundaries to construct hypothesis-driven validation
    configs that account for interactions, replacing the naive
    "apply calibrated best values" approach.
    """

    def __init__(self, results_dir: Path, log_fn=None):
        self.results_dir = Path(results_dir)
        self.records: list[GenerationRecord] = []
        self.pooled_main: dict[str, PooledMainEffect] = {}
        self.pooled_interactions: dict[str, PooledInteraction] = {}
        self.log = log_fn or (lambda msg: None)

    # ------------------------------------------------------------------
    # Bootstrap from historical data
    # ------------------------------------------------------------------

    def bootstrap_from_effects_dir(self, arm_name: str):
        """Load all historical effects JSON files and reconstruct interactions.

        For each generation:
        1. Load effects, SE, factors_tested from effects JSON
        2. Try to recover raw responses from the JSON (new format) or TSV (legacy)
        3. If raw responses available, compute interaction contrasts
        4. Create GenerationRecord and add to pool
        """
        effects_dir = self.results_dir / "effects"
        if not effects_dir.exists():
            self.log(f"  EffectBank: no effects dir found at {effects_dir}")
            return

        # Load all effects files for this arm
        pattern = f"{arm_name}_e*_g*.json"
        effects_files = sorted(effects_dir.glob(pattern))
        self.log(f"  EffectBank: bootstrapping from {len(effects_files)} effects files")

        # Try to load TSV for raw response recovery
        tsv_responses = self._load_tsv_responses(arm_name)

        for fpath in effects_files:
            try:
                data = json.loads(fpath.read_text())
                epoch = data.get("epoch", 0)
                gen = data.get("generation", 0)
                arm = data.get("arm", arm_name)
                factors_tested = data.get("factors_tested", [])
                effects = data.get("effects", {})
                se = data.get("standard_error", 1.0)
                n_experiments = data.get("n_experiments", 12)

                if not factors_tested or not effects or se <= 0:
                    continue

                # Try to get raw responses
                raw_responses = data.get("responses")  # New format
                if raw_responses is None:
                    # Legacy: try TSV recovery
                    raw_responses = tsv_responses.get((epoch, gen))

                # Compute interaction contrasts if we have raw data
                interactions = {}
                interaction_se = 0.0
                if raw_responses is not None:
                    resp_arr = np.array([
                        r if r is not None and np.isfinite(r) else np.inf
                        for r in raw_responses
                    ])
                    n_factors = len(factors_tested)
                    if n_factors > 0:
                        design = generate_pb_design(n_factors)
                        # Only use the first n_runs rows of responses
                        n_runs = design.shape[0]
                        if len(resp_arr) >= n_runs:
                            interactions, interaction_se = (
                                compute_interaction_contrasts(
                                    design, resp_arr[:n_runs], factors_tested, se,
                                )
                            )

                n_crashes = data.get("n_crashes", 0)

                record = GenerationRecord(
                    epoch=epoch,
                    generation=gen,
                    arm=arm,
                    factors_tested=factors_tested,
                    effects=effects,
                    se=se,
                    responses=raw_responses if raw_responses else [],
                    n_runs=n_experiments,
                    n_crashes=n_crashes,
                    interaction_contrasts=interactions,
                    interaction_se=interaction_se,
                )
                self.records.append(record)

            except Exception as e:
                self.log(f"  EffectBank: error loading {fpath.name}: {e}")
                continue

        self._rebuild_pool()
        n_with_interactions = sum(1 for r in self.records if r.interaction_contrasts)
        self.log(f"  EffectBank: {len(self.records)} generations loaded, "
                 f"{n_with_interactions} with interaction data")
        self.log(f"  EffectBank: {len(self.pooled_main)} pooled main effects, "
                 f"{len(self.pooled_interactions)} pooled interactions")

    def _load_tsv_responses(self, arm_name: str) -> dict[tuple[int, int], list[float]]:
        """Parse TSV to recover per-run val_bpb values for legacy generations.

        Returns dict of (epoch, gen) -> list of val_bpb values in run order.
        """
        tsv_path = self.results_dir / f"{arm_name}.tsv"
        if not tsv_path.exists():
            return {}

        result: dict[tuple[int, int], list[tuple[int, float]]] = {}
        try:
            with open(tsv_path) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 6:
                        continue
                    arm = parts[1]
                    if arm != arm_name:
                        continue
                    try:
                        epoch = int(parts[2])
                        gen_str = parts[3]
                        if gen_str == "VAL":
                            continue  # Skip validation runs
                        gen = int(gen_str)
                    except (ValueError, IndexError):
                        continue

                    label = parts[4]
                    # Extract run number from label like "e0_g0_run03"
                    run_match = re.search(r'_run(\d+)', label)
                    if not run_match:
                        continue  # Skip foldover runs for now
                    run_idx = int(run_match.group(1))

                    try:
                        val_bpb = float(parts[5])
                    except ValueError:
                        val_bpb = float("inf")

                    key = (epoch, gen)
                    if key not in result:
                        result[key] = []
                    result[key].append((run_idx, val_bpb))

        except Exception as e:
            self.log(f"  EffectBank: TSV parse error: {e}")
            return {}

        # Sort by run index and extract just the val_bpb values
        final = {}
        for key, runs in result.items():
            runs.sort(key=lambda x: x[0])
            final[key] = [v for _, v in runs]
        return final

    # ------------------------------------------------------------------
    # Per-generation ingestion (live)
    # ------------------------------------------------------------------

    def ingest_generation(
        self,
        summary: dict,
        design: np.ndarray,
        responses: np.ndarray,
    ):
        """Add a new generation's data and update pooled estimates.

        Called by orchestrator after each generation completes.

        Args:
            summary: From generation_summary() in analyzer.py
            design: PB design matrix (n_runs, n_factors)
            responses: Raw val_bpb array (n_runs,)
        """
        factors_tested = summary.get("factors_tested", [])
        effects = summary.get("effects", {})
        se = summary.get("standard_error", 1.0)

        if not factors_tested or not effects or se <= 0:
            return

        # Compute interaction contrasts
        n_runs = design.shape[0]
        resp_arr = responses[:n_runs] if len(responses) >= n_runs else responses
        interactions, interaction_se = compute_interaction_contrasts(
            design, resp_arr, factors_tested, se,
        )

        record = GenerationRecord(
            epoch=summary.get("epoch", 0),
            generation=summary.get("generation", 0),
            arm=summary.get("arm", ""),
            factors_tested=factors_tested,
            effects=effects,
            se=se,
            responses=[float(r) if np.isfinite(r) else None for r in resp_arr],
            n_runs=int(n_runs),
            n_crashes=summary.get("n_crashes", 0),
            interaction_contrasts=interactions,
            interaction_se=interaction_se,
        )
        self.records.append(record)
        self._rebuild_pool()

    # ------------------------------------------------------------------
    # Pooling: precision-weighted meta-analysis
    # ------------------------------------------------------------------

    def _rebuild_pool(self):
        """Recompute all pooled estimates from stored records.

        Main effects: precision-weighted mean across all generations testing
        that factor. Weight = 1/SE^2 for each generation.

        Interactions: same approach over interaction contrasts.
        """
        self.pooled_main = {}
        self.pooled_interactions = {}

        # --- Pool main effects ---
        # Use effective_se which incorporates crash-rate inflation
        factor_data: dict[str, list[tuple[float, float]]] = {}  # name -> [(effect, effective_se)]
        for rec in self.records:
            eff_se = rec.effective_se
            for name, effect in rec.effects.items():
                if name not in factor_data:
                    factor_data[name] = []
                factor_data[name].append((effect, eff_se))

        for name, measurements in factor_data.items():
            n = len(measurements)
            if n == 0:
                continue

            # Precision-weighted meta-analysis with two outlier defenses:
            # 1. Crash-rate SE inflation (via effective_se) — down-weights
            #    generations with >25% crashes by inflating their SE 2.5x
            # 2. Winsorization at MAX_EFFECT — caps extreme effects at ±0.6
            #    to limit inf-driven artifacts while preserving real threshold
            #    effects (e.g. UNEMBEDDING_LR at 0.56 in zero-crash gens)
            sum_w = 0.0
            sum_we = 0.0
            effects_list = []
            for effect, se in measurements:
                if se <= 0:
                    continue
                effect_clamped = max(-MAX_EFFECT, min(MAX_EFFECT, effect))
                w = 1.0 / (se * se)  # se already includes MIN_SE floor + crash inflation
                sum_w += w
                sum_we += w * effect_clamped
                effects_list.append(effect_clamped)

            if sum_w <= 0 or not effects_list:
                continue

            weighted_effect = sum_we / sum_w
            pooled_se = 1.0 / math.sqrt(sum_w)
            pooled_t = abs(weighted_effect) / pooled_se if pooled_se > 0 else 0.0

            # Direction consistency
            if weighted_effect != 0:
                sign = 1 if weighted_effect > 0 else -1
                same_dir = sum(1 for e in effects_list if (e > 0) == (sign > 0))
                direction_consistency = same_dir / len(effects_list)
            else:
                direction_consistency = 0.5

            self.pooled_main[name] = PooledMainEffect(
                factor=name,
                n_generations=n,
                weighted_effect=weighted_effect,
                pooled_se=pooled_se,
                pooled_t=pooled_t,
                direction_consistency=direction_consistency,
                individual_effects=effects_list,
            )

        # --- Pool interaction contrasts ---
        interaction_data: dict[str, list[tuple[float, float]]] = {}
        # Also track conditional effects for each interaction
        conditional_data: dict[str, list[tuple[float, float]]] = {}

        for rec in self.records:
            # Use effective_se (crash-rate inflated) for interaction pooling too
            eff_se = rec.effective_se
            for key, contrast in rec.interaction_contrasts.items():
                if key not in interaction_data:
                    interaction_data[key] = []
                interaction_data[key].append((contrast, eff_se))

        for key, measurements in interaction_data.items():
            n = len(measurements)
            if n < 2:
                continue  # Need at least 2 generations for meaningful pooling

            sum_w = 0.0
            sum_wc = 0.0
            for contrast, se in measurements:
                if se <= 0:
                    continue
                w = 1.0 / (se * se)
                sum_w += w
                sum_wc += w * contrast

            if sum_w <= 0:
                continue

            weighted_contrast = sum_wc / sum_w
            pooled_se = 1.0 / math.sqrt(sum_w)
            pooled_t = abs(weighted_contrast) / pooled_se if pooled_se > 0 else 0.0

            # Parse factor names from key "A×B"
            parts = key.split("×")
            if len(parts) != 2:
                continue
            factor_a, factor_b = parts

            self.pooled_interactions[key] = PooledInteraction(
                factor_a=factor_a,
                factor_b=factor_b,
                n_generations=n,
                weighted_contrast=weighted_contrast,
                pooled_se=pooled_se,
                pooled_t=pooled_t,
            )

        # Compute conditional effects for significant interactions
        self._compute_pooled_conditionals()

    def _compute_pooled_conditionals(self):
        """Compute pooled conditional effects for significant interactions.

        For interaction A×B, we want to know:
        - effect_A|B+: how does A affect val_bpb when B is at high level?
        - effect_A|B-: how does A affect val_bpb when B is at low level?

        These are pooled across all generations where both A and B were tested.
        """
        for key, interaction in self.pooled_interactions.items():
            if interaction.pooled_t < 1.5:
                continue  # Not worth computing conditionals for weak interactions

            fa, fb = interaction.factor_a, interaction.factor_b
            cond_a_b_plus = []  # effect_A when B=+1
            cond_a_b_minus = []  # effect_A when B=-1

            for rec in self.records:
                if fa not in rec.factors_tested or fb not in rec.factors_tested:
                    continue
                if not rec.responses or rec.se <= 0:
                    continue

                # Reconstruct design
                n_factors = len(rec.factors_tested)
                design = generate_pb_design(n_factors)
                n_runs = design.shape[0]
                resp = np.array([
                    r if r is not None and np.isfinite(r) else np.inf
                    for r in rec.responses[:n_runs]
                ])

                fa_idx = rec.factors_tested.index(fa)
                fb_idx = rec.factors_tested.index(fb)

                # Conditional effect of A when B=+1
                b_plus = design[:, fb_idx] > 0
                a_plus_b_plus = (design[:, fa_idx] > 0) & b_plus
                a_minus_b_plus = (design[:, fa_idx] < 0) & b_plus

                vals_pp = resp[a_plus_b_plus]
                vals_mp = resp[a_minus_b_plus]
                vals_pp = vals_pp[np.isfinite(vals_pp)]
                vals_mp = vals_mp[np.isfinite(vals_mp)]
                if len(vals_pp) > 0 and len(vals_mp) > 0:
                    cond_a_b_plus.append(
                        (float(np.mean(vals_pp) - np.mean(vals_mp)), rec.effective_se)
                    )

                # Conditional effect of A when B=-1
                b_minus = design[:, fb_idx] < 0
                a_plus_b_minus = (design[:, fa_idx] > 0) & b_minus
                a_minus_b_minus = (design[:, fa_idx] < 0) & b_minus

                vals_pm = resp[a_plus_b_minus]
                vals_mm = resp[a_minus_b_minus]
                vals_pm = vals_pm[np.isfinite(vals_pm)]
                vals_mm = vals_mm[np.isfinite(vals_mm)]
                if len(vals_pm) > 0 and len(vals_mm) > 0:
                    cond_a_b_minus.append(
                        (float(np.mean(vals_pm) - np.mean(vals_mm)), rec.effective_se)
                    )

            # Pool conditional effects
            if cond_a_b_plus:
                sum_w = sum(1 / (se * se) for _, se in cond_a_b_plus if se > 0)
                sum_we = sum(e / (se * se) for e, se in cond_a_b_plus if se > 0)
                if sum_w > 0:
                    interaction.conditional_a_given_b_plus = sum_we / sum_w

            if cond_a_b_minus:
                sum_w = sum(1 / (se * se) for _, se in cond_a_b_minus if se > 0)
                sum_we = sum(e / (se * se) for e, se in cond_a_b_minus if se > 0)
                if sum_w > 0:
                    interaction.conditional_a_given_b_minus = sum_we / sum_w

    # ------------------------------------------------------------------
    # Config prediction
    # ------------------------------------------------------------------

    def predict_optimal_config(
        self,
        baseline: dict,
        locked: dict,
        calibrating: dict,
        screening_threshold: float = 1.5,
    ) -> dict:
        """Construct optimal config from pooled evidence + interaction model.

        Algorithm:
        1. Start from baseline (best-ever validated config)
        2. Apply locked factors
        3. For each unlocked factor with strong, consistent pooled evidence:
           apply the better direction
        4. For factors with significant interactions:
           compute conditional optimal level given partner's setting
        5. Leave ambiguous factors at baseline (conservative)

        Args:
            baseline: Best-ever validated config (starting point)
            locked: Dict of locked factor -> value
            calibrating: Dict of calibrating factor -> cal_data
            screening_threshold: t-ratio for significance

        Returns:
            Proposed config dict
        """
        config = dict(baseline)
        config.update(locked)

        # Track which factors were set by Phase 1 (don't override in Phase 2)
        phase1_set = set()
        # Track which factors were set by Phase 2 interaction logic
        interaction_adjusted = set()

        # Phase 1: Unambiguous main effects (high consistency, significant)
        for name, pool in self.pooled_main.items():
            if name in locked:
                continue
            if pool.pooled_t < screening_threshold:
                continue
            if pool.direction_consistency < 0.70:
                continue  # Ambiguous — might be interaction, handle in Phase 2

            # Strong, consistent signal → apply better direction
            if name in calibrating:
                cal = calibrating[name]
                if pool.weighted_effect < 0:
                    # High level better (lower val_bpb)
                    config[name] = cal["range_high"]
                else:
                    # Low level better
                    config[name] = cal["range_low"]
            else:
                f = find_factor(name)
                if f:
                    if pool.weighted_effect < 0:
                        config[name] = f.high
                    else:
                        config[name] = f.low
            phase1_set.add(name)

        # Phase 2: Interaction-aware adjustments
        # TARGETED: Only adjust direction-inconsistent factors (where the main
        # effect is ambiguous due to interactions). Requires strong interaction
        # evidence (t > 5.0) to avoid false-positive-driven config changes.
        # Phase 1-set factors are never overridden.
        inconsistent_names = {
            p.factor for p in self.get_direction_inconsistent_factors(min_gens=4)
        }
        interaction_threshold = max(5.0, screening_threshold * 3)
        sig_interactions = self.get_significant_interactions(
            threshold=interaction_threshold
        )

        for interaction in sig_interactions:
            fa = interaction.factor_a
            fb = interaction.factor_b

            # Determine current level of each factor in config
            for a_name, b_name in [(fa, fb), (fb, fa)]:
                if b_name in phase1_set or b_name in interaction_adjusted:
                    continue  # Don't override strong Phase 1 or earlier interaction
                if b_name not in inconsistent_names:
                    continue  # Only adjust direction-inconsistent factors
                if a_name in locked and b_name not in locked:
                    # a_name is fixed → adjust b_name conditionally
                    a_val = config.get(a_name)
                    a_factor = find_factor(a_name)
                    if a_val is None or a_factor is None:
                        continue

                    # Determine if a is at "high" or "low" level
                    a_mid = (a_factor.high + a_factor.low) / 2
                    a_is_high = a_val >= a_mid

                    # Get conditional effect of b given a's level
                    key = f"{b_name}×{a_name}"
                    rev_key = f"{a_name}×{b_name}"

                    cond_effect = None
                    if key in self.pooled_interactions:
                        ix = self.pooled_interactions[key]
                        cond_effect = (ix.conditional_a_given_b_plus if a_is_high
                                       else ix.conditional_a_given_b_minus)
                    elif rev_key in self.pooled_interactions:
                        ix = self.pooled_interactions[rev_key]
                        # Swap: conditional of A given B
                        cond_effect = (ix.conditional_a_given_b_plus if a_is_high
                                       else ix.conditional_a_given_b_minus)

                    if cond_effect is not None and abs(cond_effect) > 0:
                        b_factor = find_factor(b_name)
                        if b_factor:
                            if b_name in calibrating:
                                cal = calibrating[b_name]
                                if cond_effect < 0:
                                    config[b_name] = cal["range_high"]
                                else:
                                    config[b_name] = cal["range_low"]
                            else:
                                if cond_effect < 0:
                                    config[b_name] = b_factor.high
                                else:
                                    config[b_name] = b_factor.low
                            interaction_adjusted.add(b_name)

        # Phase 3: Conservative fallback for ambiguous factors
        for name, pool in self.pooled_main.items():
            if name in locked or name in interaction_adjusted:
                continue
            if pool.direction_consistency < 0.55 and pool.n_generations >= 4:
                # Direction inconsistent → interaction signal, keep at baseline
                config[name] = baseline.get(name, config.get(name))

        return config

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_significant_interactions(
        self, threshold: float = 1.5,
    ) -> list[PooledInteraction]:
        """Return interactions with pooled_t > threshold, sorted by magnitude."""
        sig = [
            ix for ix in self.pooled_interactions.values()
            if ix.pooled_t > threshold
        ]
        sig.sort(key=lambda x: x.pooled_t, reverse=True)
        return sig

    def get_direction_inconsistent_factors(
        self, min_gens: int = 4,
    ) -> list[PooledMainEffect]:
        """Return factors with direction_consistency < 0.55 (interaction signal)."""
        return [
            pool for pool in self.pooled_main.values()
            if pool.n_generations >= min_gens and pool.direction_consistency < 0.55
        ]

    # ------------------------------------------------------------------
    # Logging / serialization
    # ------------------------------------------------------------------

    def summary_log(self) -> str:
        """Human-readable summary for orchestrator log."""
        lines = [f"\n--- EFFECT BANK SUMMARY ({len(self.records)} generations) ---"]

        # Top main effects
        sorted_main = sorted(
            self.pooled_main.values(),
            key=lambda x: x.pooled_t,
            reverse=True,
        )
        lines.append(f"  Pooled main effects ({len(sorted_main)} factors):")
        for pool in sorted_main[:15]:
            direction = "better@high" if pool.weighted_effect < 0 else "better@low"
            consistency = f"{pool.direction_consistency:.0%}"
            sig_marker = "***" if pool.pooled_t > 1.5 else "   "
            lines.append(
                f"    {sig_marker} {pool.factor}: pooled_t={pool.pooled_t:.2f} "
                f"({direction}, {consistency} consistent, n={pool.n_generations})"
            )

        # Direction-inconsistent factors (interaction signals)
        inconsistent = self.get_direction_inconsistent_factors()
        if inconsistent:
            lines.append(f"  Direction-inconsistent factors (interaction signals):")
            for pool in inconsistent:
                lines.append(
                    f"    ⚡ {pool.factor}: consistency={pool.direction_consistency:.0%} "
                    f"(n={pool.n_generations})"
                )

        # Significant interactions
        sig_ix = self.get_significant_interactions(threshold=1.5)
        if sig_ix:
            lines.append(f"  Significant interactions ({len(sig_ix)}):")
            for ix in sig_ix[:10]:
                lines.append(
                    f"    {ix.factor_a}×{ix.factor_b}: pooled_t={ix.pooled_t:.2f} "
                    f"contrast={ix.weighted_contrast:.6f} (n={ix.n_generations})"
                )
                if abs(ix.conditional_a_given_b_plus) > 0:
                    lines.append(
                        f"      {ix.factor_a}|{ix.factor_b}+: {ix.conditional_a_given_b_plus:.6f}  "
                        f"{ix.factor_a}|{ix.factor_b}-: {ix.conditional_a_given_b_minus:.6f}"
                    )

        lines.append("--- END EFFECT BANK ---")
        return "\n".join(lines)

    def to_json(self) -> dict:
        """JSON-serializable snapshot for persistence."""
        return {
            "n_records": len(self.records),
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pooled_main_effects": {
                name: {
                    "n_generations": p.n_generations,
                    "weighted_effect": p.weighted_effect,
                    "pooled_se": p.pooled_se,
                    "pooled_t": p.pooled_t,
                    "direction_consistency": p.direction_consistency,
                }
                for name, p in self.pooled_main.items()
            },
            "pooled_interactions": {
                key: {
                    "factor_a": ix.factor_a,
                    "factor_b": ix.factor_b,
                    "n_generations": ix.n_generations,
                    "weighted_contrast": ix.weighted_contrast,
                    "pooled_se": ix.pooled_se,
                    "pooled_t": ix.pooled_t,
                    "conditional_a_given_b_plus": ix.conditional_a_given_b_plus,
                    "conditional_a_given_b_minus": ix.conditional_a_given_b_minus,
                }
                for key, ix in self.pooled_interactions.items()
            },
        }

    def save(self, path: Path | None = None):
        """Persist the effect bank snapshot to JSON."""
        if path is None:
            path = self.results_dir / "effect_bank.json"
        path.write_text(json.dumps(self.to_json(), indent=2))


# ---------------------------------------------------------------------------
# Standalone interaction contrast computation
# ---------------------------------------------------------------------------

def compute_interaction_contrasts(
    design: np.ndarray,
    responses: np.ndarray,
    factor_names: list[str],
    se: float,
) -> tuple[dict[str, float], float]:
    """
    Compute all pairwise interaction contrasts within one PB generation.

    For factors A, B in a PB(12,11) design, the 12 rows partition into
    4 quadrants: (A+B+, A+B-, A-B+, A-B-), each with 3 rows.

    interaction_AB = (mean(A+B+ ∪ A-B-) - mean(A+B- ∪ A-B+)) / 2

    A negative contrast means "same-direction levels" are better (synergy):
    both at high or both at low beats mixed levels.

    Only stores contrasts with |contrast| > 0.5 * SE to reduce noise.

    Args:
        design: PB design matrix (n_runs, n_factors)
        responses: val_bpb array (n_runs,)
        factor_names: Factor name for each column
        se: Lenth's PSE from main effect analysis

    Returns:
        (dict of "A×B" -> contrast, interaction_se)
    """
    n_runs, n_factors = design.shape
    if len(responses) < n_runs:
        return {}, 0.0

    resp = responses[:n_runs].copy()
    # Mask non-finite values
    valid = np.isfinite(resp)

    contrasts = {}
    noise_threshold = 0.5 * se if se > 0 else 0.0

    for i in range(n_factors):
        for j in range(i + 1, n_factors):
            # Quadrant masks
            a_plus = design[:, i] > 0
            a_minus = design[:, i] < 0
            b_plus = design[:, j] > 0
            b_minus = design[:, j] < 0

            # Same-direction quadrants (++, --)
            same = (a_plus & b_plus) | (a_minus & b_minus)
            same_valid = same & valid
            # Opposite-direction quadrants (+-, -+)
            diff = (a_plus & b_minus) | (a_minus & b_plus)
            diff_valid = diff & valid

            n_same = same_valid.sum()
            n_diff = diff_valid.sum()

            if n_same < 2 or n_diff < 2:
                continue  # Not enough data

            mean_same = float(np.mean(resp[same_valid]))
            mean_diff = float(np.mean(resp[diff_valid]))
            contrast = (mean_same - mean_diff) / 2.0

            if abs(contrast) > noise_threshold:
                key = f"{factor_names[i]}×{factor_names[j]}"
                contrasts[key] = contrast

    # SE of interaction contrasts ≈ SE of main effects
    # In PB(12,11), each quadrant has 3 rows, so variance is similar
    interaction_se = se  # Approximation; could be refined

    return contrasts, interaction_se
