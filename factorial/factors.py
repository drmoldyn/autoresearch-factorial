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

    # New v2 factors — untested architectural/optimizer choices
    Factor("KV_HEAD_RATIO", low=0, high=1, baseline=1, dtype="categorical",
           apply_mode="code:kv_head_ratio"),
    # 0 = MQA (n_kv_head=1), 1 = full MHA (n_kv_head=n_head)
    Factor("CAUTIOUS_WD", low=0, high=1, baseline=1, dtype="categorical",
           apply_mode="code:cautious_wd"),
    # 0 = standard WD, 1 = cautious WD (mask by gradient-param alignment)
    Factor("RESID_LR_RATIO", low=0.001, high=0.1, baseline=0.01),
    # Multiplier on SCALAR_LR for resid_lambdas

    # Formerly locked continuous — now unlocked for calibration
    Factor("LOGIT_CAP", low=15, high=45, baseline=30),
    Factor("EMBED_WD", low=0.0, high=0.005, baseline=0.0),

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
    """Model dim must not exceed 384 (larger can't converge in 5-min budget).

    With DEPTH=4, ASPECT_RATIO=64, model_dim is quantized by HEAD_DIM:
      HD=128 → 256 (baseline), HD=160 → 320, HD=176 → 352, HD=192 → 384.
      HD=224 → 448 (too big: 25.6GB VRAM, val_bpb > 2.0, wastes compute).
    Cap at 384 to prevent outlier-dominated generations (SE inflation 20x).
    """
    depth = int(cfg.get("DEPTH", 4))
    aspect = int(cfg.get("ASPECT_RATIO", 64))
    head_dim = int(cfg.get("HEAD_DIM", 128))
    model_dim = ((depth * aspect + head_dim - 1) // head_dim) * head_dim
    return model_dim <= 384


def _valid_mlp_hidden_even(cfg: dict) -> bool:
    """MLP hidden dimension must be even for efficient computation."""
    depth = int(cfg.get("DEPTH", 4))
    aspect = int(cfg.get("ASPECT_RATIO", 64))
    head_dim = int(cfg.get("HEAD_DIM", 128))
    model_dim = ((depth * aspect + head_dim - 1) // head_dim) * head_dim
    expansion = float(cfg.get("MLP_EXPANSION", 4.0))
    hidden = int(expansion * model_dim)
    return hidden % 2 == 0


def _valid_head_dim_even(cfg: dict) -> bool:
    """HEAD_DIM must be even (RoPE positional encoding requires even dims)."""
    head_dim = int(cfg.get("HEAD_DIM", 128))
    return head_dim % 2 == 0


CONSTRAINTS = [_valid_grad_accum, _valid_model_dim, _valid_model_dim_upper_bound,
               _valid_mlp_hidden_even, _valid_head_dim_even]


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

    Priority order:
    0. CALIBRATING factors (always included — refined range from directed search)
    1. Untested rotation factors
    2. Medium-confidence factors (re-screen at refined ranges)
    3. Low-confidence rotation factors
    4. Fill with unlocked epoch-0 factors

    Calibrating factors get priority because they're proven-significant
    continuous factors undergoing directed range refinement. They must stay
    in the design to detect interaction effects and continue converging.
    """
    available = []
    used_names = set()

    # 0. Calibrating factors (always included, with knowledge-driven ranges)
    if knowledge_store:
        cal_factors = knowledge_store.get_calibrating_factors()
        for name, cal_data in cal_factors.items():
            if name in locked_factors or name in used_names:
                continue
            cf = make_calibrated_factor(name, cal_data)
            if cf is not None:
                available.append(cf)
                used_names.add(name)

    if epoch == 0:
        # Epoch 0: calibrating factors + epoch-0 factors
        for f in EPOCH_0_FACTORS:
            if f.name not in locked_factors and f.name not in used_names:
                available.append(f)
                used_names.add(f.name)
        return available[:19]

    # Epoch 1+: calibrating + rotation priority

    # 1. Untested rotation factors (highest priority for new epochs)
    for f in ROTATION_FACTORS:
        if f.name in locked_factors or f.name in used_names:
            continue
        # Skip if parent dependency is locked at wrong level
        if f.depends_on:
            parent, req = f.depends_on
            if parent in locked_factors:
                # Can't know the value here, include and let fix_config handle it
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
            if confidence in ("high", "locked", "calibrating"):
                continue  # Calibrating already included above
        available.append(f)
        used_names.add(f.name)

    # 4. Fill remaining slots with unlocked, non-high-confidence epoch-0 factors
    for f in EPOCH_0_FACTORS:
        if f.name in locked_factors or f.name in used_names:
            continue
        if knowledge_store:
            confidence = knowledge_store.get_factor_confidence(f.name)
            if confidence in ("high", "locked", "calibrating"):
                continue
        available.append(f)
        used_names.add(f.name)

    # Limit to 19 factors (20-run PB design)
    return available[:19]


# ---------------------------------------------------------------------------
# Hard physical limits for calibration-mode directed search.
# Initial factor.low/high are the *starting* search region; calibration
# can explore beyond these up to FACTOR_BOUNDS (true physical limits).
# Factors not listed here use their initial low/high as hard bounds.
# ---------------------------------------------------------------------------

FACTOR_BOUNDS = {
    # Optimizer
    "ADAM_BETA1": (0.5, 0.999),
    "ADAM_BETA2": (0.5, 0.999),
    "EMBEDDING_LR": (0.01, 2.0),
    "WEIGHT_DECAY": (0.0, 1.0),
    "MATRIX_LR": (0.001, 0.5),
    "SCALAR_LR": (0.01, 2.0),
    "UNEMBEDDING_LR": (0.0001, 0.05),
    "RESID_LR_RATIO": (0.0001, 1.0),
    # Schedule
    "WARMDOWN_RATIO": (0.0, 1.0),
    "WARMUP_RATIO": (0.0, 0.3),
    "FINAL_LR_FRAC": (0.0, 0.5),
    "X0_LAMBDA_INIT": (0.001, 1.0),
    # Architecture
    "TOTAL_BATCH_SIZE_EXP": (14, 18),  # min 14 = 2^14 = 16384 (DEVICE_BS=8 * seq=2048)
    "MLP_EXPANSION": (1.0, 8.0),
    "HEAD_DIM": (32, 192),  # 192 → model_dim=384 (max trainable in 5 min)
    "VE_GATE_CHANNELS": (4, 128),
    "SHORT_WINDOW_FRAC": (0.03125, 1.0),
    "LOGIT_CAP": (5, 100),
    "INIT_SCALE": (0.1, 10.0),
    # Muon
    "MUON_MOMENTUM": (0.5, 0.999),
    "MUON_BETA2": (0.5, 0.999),
    "NS_STEPS": (1, 15),
    # Regularization
    "LM_HEAD_WD": (0.0, 0.1),
    "ROPE_BASE": (1000, 1000000),
    "VE_WD": (0.0, 0.01),
    "EMBED_WD": (0.0, 0.01),
}


# ---------------------------------------------------------------------------
# Helper functions for strategy module
# ---------------------------------------------------------------------------

ALL_FACTORS = {f.name: f for f in EPOCH_0_FACTORS + ROTATION_FACTORS}


def find_factor(name: str) -> Factor | None:
    """Look up a factor by name from either EPOCH_0 or ROTATION sets."""
    return ALL_FACTORS.get(name)


def get_factor_bounds(name: str) -> tuple[float, float]:
    """Get hard physical limits for a factor (for calibration search).
    Falls back to the original factor definition if not in FACTOR_BOUNDS."""
    if name in FACTOR_BOUNDS:
        return FACTOR_BOUNDS[name]
    f = ALL_FACTORS.get(name)
    if f:
        return (f.low, f.high)
    return (float("-inf"), float("inf"))


def is_categorical(factor_or_name) -> bool:
    """Check if a factor is categorical/binary (should be locked, not calibrated).

    Binary and categorical factors have discrete levels that can't be refined
    through range narrowing. They should be fully locked when proven significant.
    Continuous and integer factors should be calibrated (range narrowing) instead.
    """
    if isinstance(factor_or_name, Factor):
        return factor_or_name.dtype == "categorical"
    # String name lookup
    f = ALL_FACTORS.get(factor_or_name)
    if f:
        return f.dtype == "categorical"
    # Known categoricals that may not be in ALL_FACTORS (e.g. locked)
    return factor_or_name in (
        "USE_MUON", "ACTIVATION", "WINDOW_PATTERN",
        "KV_HEAD_RATIO", "CAUTIOUS_WD",
    )


def compute_calibration_range(
    best_value: float,
    effect: float,
    tested_low: float,
    tested_high: float,
    name: str,
) -> tuple[float, float]:
    """
    Compute next calibration range using directed search.

    Shifts the test range toward the direction of improvement:
    - Positive effect (low level better): explore below best_value
    - Negative effect (high level better): explore above best_value
    - Near-zero effect: narrow around best_value

    The best_value stays within the new range. Range is clamped to
    FACTOR_BOUNDS (hard physical limits), which are wider than the
    initial Factor.low/high search region.

    Example trace (ADAM_BETA2):
        tested [0.85, 0.99], low better → best=0.85
        → new range [0.745, 0.885]  (shifted down, best in upper portion)
        tested [0.745, 0.885], high better → best=0.885
        → new range [0.85, 0.99]  (shifted up, converging on ~0.85-0.9)
    """
    original_low, original_high = get_factor_bounds(name)
    span = tested_high - tested_low

    if span < 1e-10:
        # Degenerate range — use 10% of original range around best
        fallback_span = (original_high - original_low) * 0.1
        return (
            max(original_low, best_value - fallback_span / 2),
            min(original_high, best_value + fallback_span / 2),
        )

    if abs(effect) < 1e-10:
        # No clear direction → narrow around best
        new_low = best_value - span * 0.3
        new_high = best_value + span * 0.3
    elif effect > 0:
        # Low level better → explore below best_value
        # best_value in upper quarter, extend range downward
        new_low = best_value - span * 0.75
        new_high = best_value + span * 0.25
    else:
        # High level better → explore above best_value
        # best_value in lower quarter, extend range upward
        new_low = best_value - span * 0.25
        new_high = best_value + span * 0.75

    # Clamp to hard physical bounds
    new_low = max(original_low, new_low)
    new_high = min(original_high, new_high)

    # Minimum span: 5% of original range (prevent degenerate narrowing)
    min_span = (original_high - original_low) * 0.05
    if new_high - new_low < min_span:
        center = best_value
        new_low = max(original_low, center - min_span / 2)
        new_high = min(original_high, center + min_span / 2)

    # Integer factors: round bounds to integers
    f = ALL_FACTORS.get(name)
    if f and f.dtype == "int":
        import math
        new_low = int(math.floor(new_low))
        new_high = int(math.ceil(new_high))
        if new_low == new_high:
            # Expand range: try up first, then down
            if new_high + 1 <= int(original_high):
                new_high += 1
            elif new_low - 1 >= int(original_low):
                new_low -= 1

    # HEAD_DIM must be even (RoPE requires even dims)
    if name == "HEAD_DIM":
        new_low = int(new_low)
        new_high = int(new_high)
        # Round low up to even, high down to even
        if new_low % 2 != 0:
            new_low += 1
        if new_high % 2 != 0:
            new_high -= 1
        # Ensure range is valid
        if new_high <= new_low:
            new_high = new_low + 2

    return round(new_low, 6), round(new_high, 6)


def make_calibrated_factor(name: str, cal_data: dict) -> Factor | None:
    """Create a Factor with calibration-refined range from knowledge store data.

    Args:
        name: Factor name.
        cal_data: Dict with keys 'best_value', 'range_low', 'range_high'.

    Returns:
        A Factor with narrowed range, or None if the original factor isn't found.
    """
    f = ALL_FACTORS.get(name)
    if f is None:
        return None
    return Factor(
        name=f.name,
        low=cal_data["range_low"],
        high=cal_data["range_high"],
        baseline=cal_data["best_value"],
        dtype=f.dtype,
        apply_mode=f.apply_mode,
        depends_on=f.depends_on,
    )


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
    Return prioritized rotation factor candidates, excluding locked, used,
    and calibrating factors (which are handled separately with priority).

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
            if conf in ("locked", "high", "calibrating"):
                continue  # Calibrating handled by get_factor_rotation step 0
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
    "KV_HEAD_RATIO": {0: "mqa", 1: "mha"},
    "CAUTIOUS_WD": {0: "standard", 1: "cautious"},
}
