"""
Factor definitions, ranges, rotation schedule, and constraint checking.

Each Factor represents a tunable hyperparameter with two levels (low/high)
for Plackett-Burman screening. Factors can have dependencies on other factors
and hard constraints that prevent invalid combinations.
"""

from dataclasses import dataclass, field


@dataclass
class Factor:
    name: str
    low: float
    high: float
    baseline: float  # current best / center value
    dtype: str = "float"  # "int", "float", "categorical"
    # How to write this factor into train.py:
    # "constant" -> replace MODULE_LEVEL_CONSTANT = value
    # "code:key" -> swap code block identified by key
    apply_mode: str = "constant"
    # If this factor depends on another factor being at a specific level:
    # ("PARENT_FACTOR", parent_value_required)
    depends_on: tuple | None = None

    def level_value(self, level: int) -> float | int | str:
        """Convert PB level (+1/-1) to actual value."""
        val = self.high if level == 1 else self.low
        if self.dtype == "int":
            return int(round(val))
        return val

    def refine_around(self, winner_value: float, shrink: float = 0.5):
        """Return a new Factor with narrowed range around winner_value."""
        half_range = (self.high - self.low) * shrink / 2
        new_low = max(self.low, winner_value - half_range)
        new_high = min(self.high, winner_value + half_range)
        # Ensure we still have a meaningful range
        if new_high - new_low < (self.high - self.low) * 0.1:
            new_low = winner_value - (self.high - self.low) * 0.05
            new_high = winner_value + (self.high - self.low) * 0.05
        return Factor(
            name=self.name, low=new_low, high=new_high,
            baseline=winner_value, dtype=self.dtype,
            apply_mode=self.apply_mode, depends_on=self.depends_on,
        )


# ---------------------------------------------------------------------------
# Epoch 0 factors: 11 factors screened in a 12-run PB design
# Baselines from validated best config (val_bpb=1.2413, 2026-03-17)
# Locked: USE_MUON=1, DEPTH=4, DEVICE_BATCH_SIZE=8, LOGIT_CAP=30,
#         EMBED_WD=0, ASPECT_RATIO=64
# ---------------------------------------------------------------------------

EPOCH_0_FACTORS = [
    # Core hyperparameters — re-screen at the new (much stronger) baseline
    # EXP=14 now safe: DBS=8 locked, 2^14/(8*2048)=1 ✓
    Factor("TOTAL_BATCH_SIZE_EXP", low=14, high=16, baseline=15, dtype="int"),
    # ^ stored as exponent: actual = 2**value
    Factor("EMBEDDING_LR", low=0.1, high=0.6, baseline=0.3),
    Factor("WEIGHT_DECAY", low=0.05, high=0.4, baseline=0.2),
    Factor("MATRIX_LR", low=0.02, high=0.08, baseline=0.05),

    # Schedule & optimizer
    Factor("WARMDOWN_RATIO", low=0.3, high=0.7, baseline=0.5),
    Factor("X0_LAMBDA_INIT", low=0.05, high=0.25, baseline=0.15),

    # Released from lock — re-test at new baseline
    Factor("ADAM_BETA2", low=0.85, high=0.99, baseline=0.9),
    Factor("WINDOW_PATTERN", low=0, high=1, baseline=0, dtype="categorical",
           apply_mode="code:window_pattern"),
    # 0 = "SL", 1 = "SSSL"
    Factor("HEAD_DIM", low=64, high=128, baseline=128, dtype="int"),

    # Architecture — significant but needs re-test at new baseline
    Factor("ACTIVATION", low=0, high=1, baseline=1, dtype="categorical",
           apply_mode="code:activation"),
    # 0 = gelu, 1 = relu_sq
    Factor("MLP_EXPANSION", low=2.66, high=4.0, baseline=4.0),
    # low nudged from 2.67: int(2.67*256)=683 (odd), int(2.66*256)=680 (even)
]

# ---------------------------------------------------------------------------
# Rotation factors for epochs 1+ (introduced after epoch 0 locks)
# ---------------------------------------------------------------------------

ROTATION_FACTORS = [
    # Muon sub-factors (conditional on USE_MUON — locked at 1, so always active)
    Factor("NS_STEPS", low=3, high=7, baseline=5, dtype="int",
           depends_on=("USE_MUON", 1)),
    Factor("MUON_MOMENTUM", low=0.80, high=0.95, baseline=0.85,
           depends_on=("USE_MUON", 1)),
    Factor("MUON_BETA2", low=0.90, high=0.999, baseline=0.97425,
           depends_on=("USE_MUON", 1)),

    # Optimizer & learning rates
    Factor("ADAM_BETA1", low=0.7, high=0.9, baseline=0.8),
    Factor("SCALAR_LR", low=0.1, high=0.5, baseline=0.3),
    Factor("UNEMBEDDING_LR", low=0.001, high=0.008, baseline=0.004),

    # Architecture refinement
    Factor("VE_GATE_CHANNELS", low=16, high=80, baseline=52, dtype="int"),
    Factor("SHORT_WINDOW_FRAC", low=0.0625, high=0.5, baseline=0.125),
    Factor("LM_HEAD_WD", low=0.0, high=0.01, baseline=0.0025),

    # Lower-priority (mostly insignificant in v1, but re-test at new baseline)
    Factor("ROPE_BASE", low=10000, high=200000, baseline=10000),
    Factor("VE_WD", low=0.0, high=0.003, baseline=0.0),
    Factor("INIT_SCALE", low=1.0, high=6.0, baseline=4.0),
    Factor("FINAL_LR_FRAC", low=0.0, high=0.15, baseline=0.0),
    Factor("WARMUP_RATIO", low=0.0, high=0.05, baseline=0.0),
]

# ---------------------------------------------------------------------------
# Dependency map: child factor → (parent factor, required parent level)
# A child factor is only meaningful when its parent is at the required level.
# ---------------------------------------------------------------------------

DEPENDENCIES = {
    "NS_STEPS": ("USE_MUON", 1),
    "MUON_MOMENTUM": ("USE_MUON", 1),
    "MUON_BETA2": ("USE_MUON", 1),
}

# ---------------------------------------------------------------------------
# Hard constraints: functions that return True if the config is VALID
# ---------------------------------------------------------------------------


def _valid_grad_accum(cfg: dict) -> bool:
    """TOTAL_BATCH_SIZE must be divisible by DEVICE_BATCH_SIZE * 2048."""
    batch_exp = cfg.get("TOTAL_BATCH_SIZE_EXP", 16)
    total_batch = 2 ** int(batch_exp)
    device_bs = int(cfg.get("DEVICE_BATCH_SIZE", 16))
    seq_len = 2048
    tokens_per_step = device_bs * seq_len
    return total_batch >= tokens_per_step and total_batch % tokens_per_step == 0


def _valid_model_dim(cfg: dict) -> bool:
    """Model dim must be positive and yield at least 1 head."""
    depth = int(cfg.get("DEPTH", 4))
    aspect = int(cfg.get("ASPECT_RATIO", 64))
    head_dim = int(cfg.get("HEAD_DIM", 128))
    model_dim = ((depth * aspect + head_dim - 1) // head_dim) * head_dim
    return model_dim >= head_dim


def _valid_model_dim_upper_bound(cfg: dict) -> bool:
    """Model dim must not exceed 256 (larger causes NaN in 5-min budget)."""
    depth = int(cfg.get("DEPTH", 4))
    aspect = int(cfg.get("ASPECT_RATIO", 64))
    head_dim = int(cfg.get("HEAD_DIM", 128))
    model_dim = ((depth * aspect + head_dim - 1) // head_dim) * head_dim
    return model_dim <= 256


def _valid_mlp_hidden_even(cfg: dict) -> bool:
    """MLP hidden dimension must be even for efficient computation."""
    depth = int(cfg.get("DEPTH", 4))
    aspect = int(cfg.get("ASPECT_RATIO", 64))
    head_dim = int(cfg.get("HEAD_DIM", 128))
    model_dim = ((depth * aspect + head_dim - 1) // head_dim) * head_dim
    expansion = float(cfg.get("MLP_EXPANSION", 4.0))
    hidden = int(expansion * model_dim)
    return hidden % 2 == 0


CONSTRAINTS = [_valid_grad_accum, _valid_model_dim, _valid_model_dim_upper_bound,
               _valid_mlp_hidden_even]


def check_constraints(cfg: dict) -> list[str]:
    """Return list of violated constraint names. Empty = valid."""
    violations = []
    for fn in CONSTRAINTS:
        try:
            if not fn(cfg):
                violations.append(fn.__name__)
        except Exception as e:
            violations.append(f"{fn.__name__}: {e}")
    return violations


def check_dependencies(factors: list[Factor], cfg: dict) -> dict:
    """
    Check factor dependencies. Returns a dict of factor_name -> replacement_value
    for factors whose parent dependency is not met. These factors should be
    fixed at their baseline value since they are meaningless in this config.
    """
    replacements = {}
    factor_map = {f.name: f for f in factors}
    for f in factors:
        if f.depends_on is not None:
            parent_name, required_level = f.depends_on
            parent_value = cfg.get(parent_name)
            if parent_value is not None and parent_value != required_level:
                replacements[f.name] = f.baseline
    return replacements


def _nudge_mlp_expansion(cfg: dict) -> tuple[float | None, str]:
    """Nudge MLP_EXPANSION slightly so hidden dim is even.

    Instead of reverting to baseline (losing the test), adjust by the
    minimum amount needed. Knowledge from 2.66 ≈ knowledge from 2.67.
    """
    depth = int(cfg.get("DEPTH", 4))
    aspect = int(cfg.get("ASPECT_RATIO", 64))
    head_dim = int(cfg.get("HEAD_DIM", 128))
    model_dim = ((depth * aspect + head_dim - 1) // head_dim) * head_dim
    if model_dim == 0:
        return None, ""

    expansion = float(cfg.get("MLP_EXPANSION", 4.0))
    hidden = int(expansion * model_dim)
    if hidden % 2 == 0:
        return None, ""  # Already valid

    # Round hidden to nearest even, recompute expansion
    hidden_even = hidden - 1 if (hidden - 1) % 2 == 0 else hidden + 1
    new_expansion = round(hidden_even / model_dim, 4)
    return new_expansion, f"MLP_EXPANSION: nudged {expansion}→{new_expansion} (hidden {hidden}→{hidden_even} even)"


def fix_config(factors: list[Factor], cfg: dict) -> tuple[dict, list[str]]:
    """
    Fix an invalid config. Prefers nudging values slightly over reverting
    to baseline, so the experimental knowledge is preserved.
    Returns (fixed_config, list_of_fixes_applied).
    """
    fixed = dict(cfg)
    fixes = []

    # Fix dependencies first
    dep_replacements = check_dependencies(factors, fixed)
    for name, replacement in dep_replacements.items():
        if fixed.get(name) != replacement:
            fixes.append(f"{name}: dependency not met, fixed to {replacement}")
            fixed[name] = replacement

    # Try targeted nudges before brute-force reversion
    nudged_exp, nudge_msg = _nudge_mlp_expansion(fixed)
    if nudged_exp is not None:
        fixed["MLP_EXPANSION"] = nudged_exp
        fixes.append(nudge_msg)

    # Fix remaining hard constraints by reverting changed factors to baseline.
    # Strategy: fix one violation at a time (greedy), re-checking after each
    # reversion. This handles multi-violation cases where no single factor
    # fixes ALL constraints, but reverting one factor reduces the violation set.
    violations = check_constraints(fixed)
    max_attempts = len(factors) * 2  # generous upper bound
    attempt = 0
    while violations and attempt < max_attempts:
        made_progress = False
        target_violation = violations[0]  # fix one constraint at a time
        for f in factors:
            if f.name in fixed and fixed[f.name] != f.baseline:
                test = dict(fixed)
                test[f.name] = f.baseline
                test_violations = check_constraints(test)
                # Accept if this reversion reduces the violation count
                # OR removes the specific target violation
                if (len(test_violations) < len(violations) or
                        target_violation not in test_violations):
                    fixed[f.name] = f.baseline
                    fixes.append(f"{f.name}: constraint {target_violation}, reverted to {f.baseline}")
                    made_progress = True
                    break
        if not made_progress:
            break  # no factor reversion helps — avoid infinite loop
        violations = check_constraints(fixed)
        attempt += 1

    return fixed, fixes


def get_factor_rotation(
    epoch: int,
    locked_factors: set[str],
    knowledge_store=None,
) -> list[Factor]:
    """
    Get factors for a given epoch, using knowledge-driven selection when available.

    Epoch 0: use EPOCH_0_FACTORS.
    Epoch 1+: prioritize untested rotation factors, then re-screen medium-confidence
    factors at refined ranges, filling remaining slots with unlocked epoch-0 factors.

    If knowledge_store is provided, uses factor history to:
    - Prioritize untested factors
    - Re-screen medium-confidence factors at narrower ranges
    - Skip factors that have been consistently insignificant
    """
    if epoch == 0:
        return [f for f in EPOCH_0_FACTORS if f.name not in locked_factors]

    available = []
    used_names = set()

    # 1. Untested rotation factors (highest priority for new epochs)
    for f in ROTATION_FACTORS:
        if f.name in locked_factors or f.name in used_names:
            continue
        # Skip if parent dependency is locked at wrong level
        if f.depends_on:
            parent, req = f.depends_on
            if parent in locked_factors:
                # Check if parent is locked at a different level than required
                # We can't know the value here, so include and let fix_config handle it
                pass
        is_untested = True
        if knowledge_store:
            confidence = knowledge_store.get_factor_confidence(f.name)
            is_untested = confidence == "untested"
        if is_untested:
            available.append(f)
            used_names.add(f.name)

    # 2. Medium-confidence factors from both sets (re-screen at refined ranges)
    if knowledge_store:
        all_factors = ROTATION_FACTORS + EPOCH_0_FACTORS
        for f in all_factors:
            if f.name in locked_factors or f.name in used_names:
                continue
            confidence = knowledge_store.get_factor_confidence(f.name)
            if confidence == "medium":
                # Refine range around best known value
                latest_effect = knowledge_store.get_latest_effect(f.name)
                if latest_effect is not None and latest_effect < 0 and f.dtype != "categorical":
                    # Effect was negative (high level better) — refine around high
                    refined = f.refine_around(f.high)
                elif latest_effect is not None and latest_effect > 0 and f.dtype != "categorical":
                    refined = f.refine_around(f.low)
                else:
                    refined = f
                available.append(refined)
                used_names.add(f.name)

    # 3. Low-confidence rotation factors (tested but not yet significant enough)
    for f in ROTATION_FACTORS:
        if f.name in locked_factors or f.name in used_names:
            continue
        if knowledge_store:
            confidence = knowledge_store.get_factor_confidence(f.name)
            if confidence in ("high", "locked"):
                continue  # Don't re-test high-confidence factors endlessly
        available.append(f)
        used_names.add(f.name)

    # 4. Fill remaining slots with unlocked, non-high-confidence epoch-0 factors
    for f in EPOCH_0_FACTORS:
        if f.name in locked_factors or f.name in used_names:
            continue
        if knowledge_store:
            confidence = knowledge_store.get_factor_confidence(f.name)
            if confidence in ("high", "locked"):
                continue
        available.append(f)
        used_names.add(f.name)

    # Limit to 19 factors (20-run PB design)
    return available[:19]


# ---------------------------------------------------------------------------
# Helper functions for strategy module
# ---------------------------------------------------------------------------

ALL_FACTORS = {f.name: f for f in EPOCH_0_FACTORS + ROTATION_FACTORS}


def find_factor(name: str) -> Factor | None:
    """Look up a factor by name from either EPOCH_0 or ROTATION sets."""
    return ALL_FACTORS.get(name)


def get_muon_subfactors() -> list[Factor]:
    """Return Muon-dependent sub-factors for conditional expansion."""
    return [f for f in ROTATION_FACTORS
            if f.depends_on and f.depends_on[0] == "USE_MUON"]


def get_rotation_candidates(
    epoch: int,
    locked: set[str],
    already_used: set[str],
    knowledge=None,
) -> list[Factor]:
    """
    Return prioritized rotation factor candidates, excluding locked and used.
    Priority: untested > medium-confidence re-test > low-confidence.
    """
    untested = []
    medium = []
    low = []

    for f in ROTATION_FACTORS + EPOCH_0_FACTORS:
        if f.name in locked or f.name in already_used:
            continue
        if knowledge:
            conf = knowledge.get_factor_confidence(f.name)
            if conf in ("locked", "high"):
                continue
            elif conf == "untested":
                untested.append(f)
            elif conf == "medium":
                latest = knowledge.get_latest_effect(f.name)
                if latest is not None and f.dtype != "categorical":
                    if latest < 0:
                        medium.append(f.refine_around(f.high))
                    else:
                        medium.append(f.refine_around(f.low))
                else:
                    medium.append(f)
            elif conf == "low":
                low.append(f)
        else:
            untested.append(f)

    return untested + medium + low


def clear_muon_dependent_locks(
    locked: dict[str, float], muon_state: int | None,
) -> dict[str, float]:
    """
    If USE_MUON changes state, clear locks on Muon-dependent sub-factors.
    These locks were established under a different Muon state and are
    now potentially misleading.
    """
    if muon_state is None:
        return locked
    cleaned = {}
    for name, value in locked.items():
        if name in DEPENDENCIES:
            parent, req = DEPENDENCIES[name]
            if parent == "USE_MUON" and muon_state != req:
                continue  # Drop this lock — Muon state changed
        cleaned[name] = value
    return cleaned


# Categorical value mappings for code-swap factors
CATEGORICAL_VALUES = {
    "WINDOW_PATTERN": {0: "SL", 1: "SSSL"},
    "ACTIVATION": {0: "gelu", 1: "relu_sq"},
    "USE_MUON": {0: False, 1: True},
}
