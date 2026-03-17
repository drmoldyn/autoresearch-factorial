"""
Experimental design matrix generation.

Generates Plackett-Burman (Resolution III) designs for factor screening,
and foldover designs for de-aliasing ambiguous effects.
"""

import numpy as np

try:
    from pyDOE2 import pbdesign
except ImportError:
    pbdesign = None

from .factors import Factor


def _hadamard_pb(n_runs: int) -> np.ndarray:
    """
    Generate a Plackett-Burman design matrix using Paley construction.
    Falls back to pyDOE2 if available, otherwise uses built-in construction
    for common sizes (4, 8, 12, 16, 20, 24).
    """
    if pbdesign is not None:
        # pyDOE2's pbdesign takes the number of factors, returns (n_runs, n_factors)
        # with n_runs = next multiple of 4 >= n_factors + 1
        n_factors = n_runs - 1
        design = pbdesign(n_factors)
        return design

    # Fallback: built-in Paley construction for common sizes
    # First row generators for PB designs (from Plackett & Burman 1946)
    generators = {
        4: [1, 1, -1],
        8: [1, 1, 1, -1, 1, -1, -1],
        12: [1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1],
        16: [1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1],
        20: [1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, -1, -1, -1],
        24: [1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, -1, -1],
    }
    if n_runs not in generators:
        raise ValueError(f"No built-in PB generator for {n_runs} runs. "
                         f"Install pyDOE2 or use n_runs in {sorted(generators.keys())}")

    gen = generators[n_runs]
    n_factors = n_runs - 1
    rows = []
    for i in range(n_factors):
        row = gen[i:] + gen[:i]
        rows.append(row[:n_factors])
    # Add the all-minus-one row
    rows.append([-1] * n_factors)
    return np.array(rows, dtype=float)


def generate_pb_design(n_factors: int) -> np.ndarray:
    """
    Generate a Plackett-Burman design matrix.

    Args:
        n_factors: Number of factors to screen.

    Returns:
        Design matrix of shape (n_runs, n_factors) with values +1/-1.
        n_runs is the smallest multiple of 4 >= n_factors + 1.
    """
    # PB designs require runs = multiple of 4 >= n_factors + 1
    n_runs = n_factors + 1
    if n_runs % 4 != 0:
        n_runs = ((n_runs // 4) + 1) * 4

    design = _hadamard_pb(n_runs)

    # Trim to requested number of factors (PB may have extra columns)
    if design.shape[1] > n_factors:
        design = design[:, :n_factors]

    return design


def generate_foldover(base_design: np.ndarray, factor_idx: int) -> np.ndarray:
    """
    Generate a foldover design for a specific factor.

    A foldover mirrors the sign of one factor column, generating complementary
    runs that de-alias the factor's main effect from its confounded 2-factor
    interactions. Only the rows where the factor was at +1 are mirrored,
    producing a half-foldover (more efficient than full foldover).

    Args:
        base_design: Original PB design matrix (n_runs, n_factors).
        factor_idx: Index of the factor to de-alias.

    Returns:
        Foldover design matrix (n_foldover_runs, n_factors).
    """
    foldover = base_design.copy()
    foldover[:, factor_idx] *= -1
    # Only keep the runs where the factor is now different from the original
    # This is all runs (full foldover), but we limit to the most informative
    # subset: runs where the target factor was +1 in the original
    mask = base_design[:, factor_idx] > 0
    return foldover[mask]


def design_to_configs(
    design: np.ndarray,
    factors: list[Factor],
    baseline: dict | None = None,
) -> list[dict]:
    """
    Convert a +1/-1 design matrix to actual hyperparameter configurations.

    Each row of the design matrix becomes a config dict mapping factor names
    to their actual values (low for -1, high for +1).

    Args:
        design: PB design matrix (n_runs, n_factors).
        factors: List of Factor objects (must match design columns).
        baseline: Optional baseline config to merge with.

    Returns:
        List of config dicts, one per design row.
    """
    assert design.shape[1] == len(factors), \
        f"Design has {design.shape[1]} columns but {len(factors)} factors"

    configs = []
    for row in design:
        cfg = dict(baseline) if baseline else {}
        for j, factor in enumerate(factors):
            level = int(row[j])  # +1 or -1
            cfg[factor.name] = factor.level_value(level)
        configs.append(cfg)
    return configs


def get_alias_structure(design: np.ndarray, factor_names: list[str]) -> dict:
    """
    Compute the alias structure of a PB design.

    For Resolution III designs, each main effect is aliased with certain
    2-factor interactions. This function identifies which interactions
    are confounded with each main effect.

    Args:
        design: PB design matrix (n_runs, n_factors).
        factor_names: Names of the factors.

    Returns:
        Dict mapping each factor name to a list of (factor_i, factor_j) tuples
        representing the 2-factor interactions aliased with it.
    """
    n_runs, n_factors = design.shape
    aliases = {name: [] for name in factor_names}

    for i in range(n_factors):
        for j in range(i + 1, n_factors):
            # The interaction column is the element-wise product
            interaction = design[:, i] * design[:, j]
            # Check correlation with each main effect
            for k in range(n_factors):
                if k == i or k == j:
                    continue
                correlation = np.abs(np.dot(design[:, k], interaction)) / n_runs
                if correlation > 0.3:  # Threshold for meaningful aliasing
                    aliases[factor_names[k]].append(
                        (factor_names[i], factor_names[j])
                    )

    return aliases
