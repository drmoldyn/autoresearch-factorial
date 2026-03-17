"""
Effect computation, factor ranking, foldover decisions, and winner selection.

Implements contrast analysis for Plackett-Burman designs, Lenth's method
for standard error estimation (no need for replicate runs), and adaptive
foldover decision logic.

Two-tier adaptive significance thresholds optimized for iterative screening:
- Screening threshold: liberal inclusion for winner configs. Self-calibrates
  from π (fraction of active factors estimated from recent generations).
  Starts at ~1.5 (π ≈ 0.25) and increases toward ~2.3 as optimization
  converges and fewer factors remain active. Box & Meyer (1986) Bayesian.
- Locking threshold: screening + 1.0 uplift. Permanently locking a factor
  is hard to reverse, requiring stronger evidence at every π level.
"""

import json
import math
from pathlib import Path

import numpy as np

from .factors import Factor

# Default thresholds, used when no knowledge data is available.
# These match Box & Meyer (1986) at π = 0.25 (25% active factors).
SCREENING_THRESHOLD = 1.5
LOCKING_THRESHOLD = 2.5


def adaptive_screening_threshold(pi: float) -> float:
    """
    Compute the screening t-ratio threshold from the active fraction π.

    Based on Box & Meyer (1986) Bayesian screening: at 50% posterior odds
    of a factor being active, the critical t-ratio is approximately:

        t = 0.81 - 0.5 * ln(π)

    This naturally adapts:
    - π = 0.30 (many active, early): t ≈ 1.41 (liberal)
    - π = 0.25 (default prior):      t ≈ 1.50
    - π = 0.15 (fewer active):       t ≈ 1.76
    - π = 0.10 (near convergence):   t ≈ 1.96
    - π = 0.05 (highly optimized):   t ≈ 2.31

    Clamped to [1.0, 2.5] to avoid extremes.
    """
    pi = max(0.05, min(0.50, pi))
    t = 0.81 - 0.5 * math.log(pi)
    return max(1.0, min(2.5, t))


def adaptive_locking_threshold(pi: float) -> float:
    """
    Compute the locking t-ratio threshold from the active fraction π.

    Locking is harder to reverse than screening, so we add a fixed uplift
    of 1.0 over the screening threshold. This ensures locking requires
    substantially more evidence than screening at every π level.

    Clamped to [2.0, 3.5].
    """
    return max(2.0, min(3.5, adaptive_screening_threshold(pi) + 1.0))


def compute_main_effects(
    design: np.ndarray,
    responses: np.ndarray,
    factor_names: list[str],
) -> dict[str, float]:
    """
    Compute main effect estimates via contrast analysis.

    For each factor, the main effect is:
        effect_i = mean(response where factor_i = +1) - mean(response where factor_i = -1)

    A negative effect means the high level produced LOWER val_bpb (better).

    Args:
        design: PB design matrix (n_runs, n_factors), values +1/-1.
        responses: val_bpb results (n_runs,). Lower is better.
        factor_names: List of factor names matching design columns.

    Returns:
        Dict of factor_name -> effect_size.
    """
    n_runs, n_factors = design.shape
    assert len(responses) == n_runs
    assert len(factor_names) == n_factors

    effects = {}
    for j, name in enumerate(factor_names):
        high_mask = design[:, j] > 0
        low_mask = design[:, j] < 0
        # Filter out crashes (inf values) from effect computation
        high_vals = responses[high_mask]
        low_vals = responses[low_mask]
        high_valid = high_vals[np.isfinite(high_vals)]
        low_valid = low_vals[np.isfinite(low_vals)]

        if len(high_valid) == 0 or len(low_valid) == 0:
            effects[name] = 0.0  # Can't estimate effect
        else:
            effects[name] = float(np.mean(high_valid) - np.mean(low_valid))

    return effects


def compute_standard_error_lenth(effects: dict[str, float]) -> float:
    """
    Estimate standard error using Lenth's method.

    Lenth's method uses the median of absolute effects to estimate the
    pseudo standard error (PSE), assuming most effects are negligible.
    This avoids needing replicate runs.

    PSE = 1.5 * median(|effects| where |effect| < 2.5 * s0)
    where s0 = 1.5 * median(|all effects|)

    Args:
        effects: Dict of factor_name -> effect_size.

    Returns:
        Pseudo standard error estimate.
    """
    abs_effects = np.array([abs(e) for e in effects.values()])
    if len(abs_effects) == 0:
        return 1.0  # Fallback

    # Initial estimate
    s0 = 1.5 * float(np.median(abs_effects))
    if s0 < 1e-10:
        return float(np.mean(abs_effects)) if np.mean(abs_effects) > 0 else 1.0

    # Iterative trimming
    trimmed = abs_effects[abs_effects < 2.5 * s0]
    if len(trimmed) == 0:
        return s0

    pse = 1.5 * float(np.median(trimmed))
    return max(pse, 1e-10)


def rank_factors(
    effects: dict[str, float],
    se: float,
    screening_threshold: float = 1.5,
) -> list[tuple[str, float, float, bool]]:
    """
    Rank factors by effect magnitude relative to standard error.

    Uses a two-tier threshold system optimized for iterative screening:
    - screening_threshold (default 1.5): liberal inclusion for winner selection.
      False positives are self-correcting (won't replicate in next generation).
      False negatives are costly (factor dismissed, potentially never re-tested).
      Based on Box & Meyer (1986) Bayesian screening with prior π ≈ 0.25.
    - Locking uses a stricter threshold (in knowledge.py) since it's harder to reverse.

    Args:
        effects: Dict of factor_name -> effect_size.
        se: Pseudo standard error from Lenth's method.
        screening_threshold: t-ratio cutoff for significance (default 1.5).

    Returns:
        List of (name, effect, t_ratio, is_significant), sorted by |t_ratio| descending.
    """
    ranked = []
    for name, effect in effects.items():
        t_ratio = abs(effect) / se if se > 0 else 0.0
        significant = t_ratio > screening_threshold
        ranked.append((name, effect, t_ratio, significant))

    ranked.sort(key=lambda x: abs(x[2]), reverse=True)
    return ranked


def decide_foldovers(
    effects: dict[str, float],
    se: float,
    alias_structure: dict[str, list[tuple[str, str]]],
    max_foldovers: int = 4,
    screening_threshold: float = 1.5,
) -> list[str]:
    """
    Decide which factors need foldover runs to de-alias.

    A factor needs foldover when:
    1. Its effect crosses the screening threshold (|effect| > screening_threshold * SE)
    2. It has aliases with other factors that also cross the threshold
    3. The effect isn't so dominant that aliasing doesn't matter (|effect| < 3*SE)

    Foldovers invest extra runs to resolve ambiguity in the "interesting middle
    ground" — effects that are probably real but could be partially aliased
    interactions. Clearly dominant effects (t > 3) don't need foldover because
    even after de-aliasing, the factor would still be significant.

    Args:
        effects: Main effect estimates.
        se: Standard error.
        alias_structure: From get_alias_structure().
        max_foldovers: Maximum number of foldover factors.
        screening_threshold: t-ratio cutoff matching the screening threshold.

    Returns:
        List of factor names needing foldover, sorted by ambiguity.
    """
    candidates = []

    for name, effect in effects.items():
        t_ratio = abs(effect) / se if se > 0 else 0.0
        if t_ratio < screening_threshold:
            continue  # Below screening threshold, no foldover needed

        if t_ratio > 3.0:
            continue  # Clearly dominant, no foldover needed

        # Check if any aliased interaction partners also cross the threshold
        aliases = alias_structure.get(name, [])
        for partner_a, partner_b in aliases:
            partner_effect_a = abs(effects.get(partner_a, 0))
            partner_effect_b = abs(effects.get(partner_b, 0))
            partner_t_a = partner_effect_a / se if se > 0 else 0.0
            partner_t_b = partner_effect_b / se if se > 0 else 0.0
            if partner_t_a > screening_threshold or partner_t_b > screening_threshold:
                ambiguity = min(t_ratio, max(partner_t_a, partner_t_b))
                candidates.append((name, ambiguity))
                break

    # Sort by ambiguity (highest first = most urgent to resolve)
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in candidates[:max_foldovers]]


def select_winner(
    configs: list[dict],
    responses: np.ndarray,
    effects: dict[str, float],
    se: float,
    baseline: dict,
    screening_threshold: float = 1.5,
) -> dict:
    """
    Select the winning configuration from a PB generation.

    Strategy: construct the winner from the BEST LEVELS of significant factors
    rather than just picking the best observed run. This is more robust because
    PB rows are designed for effect estimation, not for containing the best
    overall config.

    The winner config = baseline, with significant factors set to their
    better level (the level that produced lower val_bpb).

    Uses a liberal screening threshold (default 1.5) because false positives
    are self-correcting in subsequent generations while false negatives
    mean permanently losing a potentially useful factor.

    Args:
        configs: List of config dicts from the design matrix.
        responses: val_bpb for each config.
        effects: Main effect estimates.
        se: Standard error.
        baseline: Current baseline configuration.
        screening_threshold: t-ratio cutoff (default 1.5).

    Returns:
        Winner configuration dict.
    """
    winner = dict(baseline)

    for name, effect in effects.items():
        t_ratio = abs(effect) / se if se > 0 else 0.0
        if t_ratio < screening_threshold:
            continue  # Below threshold, keep baseline

        # Negative effect means high level was better (lower val_bpb)
        # Positive effect means low level was better
        # Find the factor in configs to get its levels
        for cfg in configs:
            if name in cfg:
                # Check: did any config use this factor?
                # Effect < 0 means high > low (high is better for val_bpb)
                # Since lower val_bpb is better, negative effect = high is better
                if effect < 0:
                    # High level produced lower val_bpb -> use high
                    winner[name] = max(cfg[name] for cfg in configs if name in cfg)
                else:
                    # Low level produced lower val_bpb -> use low
                    winner[name] = min(cfg[name] for cfg in configs if name in cfg)
                break

    # Also record the best observed val_bpb as a sanity check
    best_idx = int(np.argmin(responses[np.isfinite(responses)])) if np.any(np.isfinite(responses)) else 0
    best_observed_bpb = float(responses[best_idx]) if np.isfinite(responses[best_idx]) else float('inf')

    return winner


def recompute_with_foldover(
    original_design: np.ndarray,
    foldover_design: np.ndarray,
    original_responses: np.ndarray,
    foldover_responses: np.ndarray,
    factor_idx: int,
    factor_names: list[str],
) -> dict[str, float]:
    """
    Recompute effects using combined original + foldover data.

    The foldover for factor_idx de-aliases its main effect from confounded
    interactions. The combined analysis uses all data for a more accurate
    estimate of this factor's true main effect.

    Args:
        original_design: Original PB design (n, k).
        foldover_design: Foldover runs (m, k).
        original_responses: Responses for original design.
        foldover_responses: Responses for foldover runs.
        factor_idx: Which factor was folded over.
        factor_names: Factor names.

    Returns:
        Updated effect estimates.
    """
    # Combine designs and responses
    combined_design = np.vstack([original_design, foldover_design])
    combined_responses = np.concatenate([original_responses, foldover_responses])

    # Recompute all effects with the augmented data
    return compute_main_effects(combined_design, combined_responses, factor_names)


def generation_summary(
    epoch: int,
    gen: int,
    arm_name: str,
    factors: list[Factor],
    effects: dict[str, float],
    se: float,
    ranked: list[tuple[str, float, float, bool]],
    winner: dict,
    responses: np.ndarray,
    foldover_factors: list[str],
) -> dict:
    """
    Create a JSON-serializable summary of a generation's results.
    """
    return {
        "epoch": epoch,
        "generation": gen,
        "arm": arm_name,
        "n_experiments": len(responses),
        "n_crashes": int(np.sum(~np.isfinite(responses))),
        "best_val_bpb": float(np.min(responses[np.isfinite(responses)])) if np.any(np.isfinite(responses)) else None,
        "worst_val_bpb": float(np.max(responses[np.isfinite(responses)])) if np.any(np.isfinite(responses)) else None,
        "median_val_bpb": float(np.median(responses[np.isfinite(responses)])) if np.any(np.isfinite(responses)) else None,
        "standard_error": se,
        "factors_tested": [f.name for f in factors],
        "effects": effects,
        "ranked_factors": [
            {"name": name, "effect": effect, "t_ratio": t, "significant": sig}
            for name, effect, t, sig in ranked
        ],
        "significant_factors": [name for name, _, _, sig in ranked if sig],
        "foldovers_run": foldover_factors,
        "winner_config": {k: v for k, v in winner.items()
                          if k in {f.name for f in factors}},
    }
