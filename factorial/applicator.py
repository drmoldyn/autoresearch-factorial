"""
Configuration applicator: writes factor levels into train.py.

Reads and modifies the hyperparameter constants at the top of train.py
based on a configuration dict. Handles both simple constant replacement
and code-block swaps for categorical factors (activation function, etc.).
"""

import re
from pathlib import Path

from .factors import CATEGORICAL_VALUES


# Mapping from factor names to the constant names in train.py
FACTOR_TO_CONSTANT = {
    "DEPTH": "DEPTH",
    "TOTAL_BATCH_SIZE_EXP": None,  # Special: TOTAL_BATCH_SIZE = 2**{value}
    "EMBEDDING_LR": "EMBEDDING_LR",
    "WEIGHT_DECAY": "WEIGHT_DECAY",
    "MATRIX_LR": "MATRIX_LR",
    "WARMDOWN_RATIO": "WARMDOWN_RATIO",
    "ROPE_BASE": None,  # Special: inside RoPE constructor
    "EMBED_WD": None,  # Special: weight decay applied to embeddings
    "VE_WD": None,  # Special: weight decay applied to value embeddings
    "X0_LAMBDA_INIT": None,  # Special: in init_weights
    "INIT_SCALE": None,  # Special: init scale multiplier
    "UNEMBEDDING_LR": "UNEMBEDDING_LR",
    "ADAM_BETA1": None,  # Special: part of ADAM_BETAS tuple
    "ADAM_BETA2": None,  # Special: part of ADAM_BETAS tuple
    "WINDOW_PATTERN": "WINDOW_PATTERN",
    "DEVICE_BATCH_SIZE": "DEVICE_BATCH_SIZE",
    "FINAL_LR_FRAC": "FINAL_LR_FRAC",
    "WARMUP_RATIO": "WARMUP_RATIO",
    "SCALAR_LR": "SCALAR_LR",
    "USE_MUON": "USE_MUON",
    "NS_STEPS": "NS_STEPS",
    "MUON_MOMENTUM": "MUON_MOMENTUM",
    "MUON_BETA2": "MUON_BETA2",
    "LOGIT_CAP": None,  # Special: softcap value in forward()
    "VE_GATE_CHANNELS": None,  # Special: in CausalSelfAttention
    "HEAD_DIM": "HEAD_DIM",
    "ASPECT_RATIO": "ASPECT_RATIO",
    "SHORT_WINDOW_FRAC": None,  # Special: short_window calculation
    "LM_HEAD_WD": None,  # Special: weight decay for lm_head
    "MLP_EXPANSION": None,  # Special: MLP hidden dim ratio
    "ACTIVATION": None,  # Special: code block swap
}


def _replace_constant(content: str, name: str, value) -> str:
    """Replace a module-level constant assignment: NAME = old -> NAME = new."""
    pattern = rf'^({re.escape(name)}\s*=\s*)(.+)$'
    if isinstance(value, bool):
        val_str = str(value)
    elif isinstance(value, int):
        val_str = str(value)
    elif isinstance(value, float):
        val_str = f"{value:.10g}"
    elif isinstance(value, str):
        val_str = f'"{value}"'
    else:
        val_str = str(value)

    new_content, count = re.subn(pattern, rf'\g<1>{val_str}', content, flags=re.MULTILINE)
    if count == 0:
        raise ValueError(f"Could not find constant '{name}' in train.py")
    return new_content


def _replace_adam_betas(content: str, beta1: float, beta2: float) -> str:
    """Replace ADAM_BETAS = (b1, b2) tuple."""
    pattern = r'^(ADAM_BETAS\s*=\s*)\(.+?\)$'
    replacement = rf'\g<1>({beta1:.10g}, {beta2:.10g})'
    new_content, count = re.subn(pattern, replacement, content, flags=re.MULTILINE)
    if count == 0:
        raise ValueError("Could not find ADAM_BETAS in train.py")
    return new_content


def _replace_total_batch_size(content: str, exponent: int) -> str:
    """Replace TOTAL_BATCH_SIZE = 2**N."""
    pattern = r'^(TOTAL_BATCH_SIZE\s*=\s*).+$'
    replacement = rf'\g<1>2**{exponent}'
    new_content, count = re.subn(pattern, replacement, content, flags=re.MULTILINE)
    if count == 0:
        raise ValueError("Could not find TOTAL_BATCH_SIZE in train.py")
    return new_content


def _replace_window_pattern(content: str, value) -> str:
    """Replace WINDOW_PATTERN string."""
    if isinstance(value, (int, float)):
        mapping = CATEGORICAL_VALUES.get("WINDOW_PATTERN", {})
        value = mapping.get(int(value), "SSSL")
    return _replace_constant(content, "WINDOW_PATTERN", value)


# --- Code-swap handlers for inline factors ---

def _replace_rope_base(content: str, value: float) -> str:
    """Replace base=NNNNN in nn.RoPE(..., base=NNNNN)."""
    pattern = r'(nn\.RoPE\([^)]*base=)\s*[\d.]+(\s*\))'
    replacement = rf'\g<1>{value:.10g}\2'
    new_content, count = re.subn(pattern, replacement, content)
    if count == 0:
        raise ValueError("Could not find nn.RoPE base= in train.py")
    return new_content


def _replace_embed_wd(content: str, value: float) -> str:
    """Replace weight_decay in the wte (embedding) optimizer config."""
    # Match: elif "wte" in path: ... "weight_decay": 0.0,
    pattern = (r'("wte" in path[\s\S]*?"weight_decay":\s*)([\d.]+)')
    new_content, count = re.subn(pattern, rf'\g<1>{value:.10g}', content, count=1)
    if count == 0:
        raise ValueError("Could not find wte weight_decay in train.py")
    return new_content


def _replace_ve_wd(content: str, value: float) -> str:
    """Replace weight_decay in the value_embeds optimizer config."""
    pattern = (r'("value_embeds" in path[\s\S]*?"weight_decay":\s*)([\d.]+)')
    new_content, count = re.subn(pattern, rf'\g<1>{value:.10g}', content, count=1)
    if count == 0:
        raise ValueError("Could not find value_embeds weight_decay in train.py")
    return new_content


def _replace_x0_lambda_init(content: str, value: float) -> str:
    """Replace the x0_lambdas initialization value."""
    pattern = r'(mx\.full\(\(self\.config\.n_layer,\),\s*)([\d.]+)'
    replacement = rf'\g<1>{value:.10g}'
    new_content, count = re.subn(pattern, replacement, content)
    if count == 0:
        raise ValueError("Could not find x0_lambdas init value in train.py")
    return new_content


def _replace_init_scale(content: str, value: float) -> str:
    """Replace the init scale factor: scale = N**0.5 * n_embd**-0.5."""
    pattern = r'(scale\s*=\s*)([\d.]+)(\*\*0\.5\s*\*\s*n_embd\*\*-0\.5)'
    replacement = rf'\g<1>{value:.10g}\3'
    new_content, count = re.subn(pattern, replacement, content)
    if count == 0:
        raise ValueError("Could not find init scale factor in train.py")
    return new_content


def _replace_logit_cap(content: str, value: float) -> str:
    """Replace logit soft-capping: N * mx.tanh(logits / N)."""
    if value <= 0:
        # Disable logit capping: noop assignment preserves logits variable
        pattern = r'(\s+)logits\s*=\s*[\d.]+\s*\*\s*mx\.tanh\(logits\s*/\s*[\d.]+\)'
        replacement = r'\1logits = logits  # logit capping disabled'
        new_content, count = re.subn(pattern, replacement, content)
    else:
        pattern = r'(logits\s*=\s*)([\d.]+)(\s*\*\s*mx\.tanh\(logits\s*/\s*)([\d.]+)(\))'
        replacement = rf'\g<1>{value:.10g}\g<3>{value:.10g}\g<5>'
        new_content, count = re.subn(pattern, replacement, content)
    if count == 0:
        raise ValueError("Could not find logit cap in train.py")
    return new_content


def _replace_ve_gate_channels(content: str, value: int) -> str:
    """Replace self.ve_gate_channels = N."""
    pattern = r'(self\.ve_gate_channels\s*=\s*)\d+'
    replacement = rf'\g<1>{int(value)}'
    new_content, count = re.subn(pattern, replacement, content)
    if count == 0:
        raise ValueError("Could not find ve_gate_channels in train.py")
    return new_content


def _replace_short_window_frac(content: str, value: float) -> str:
    """Replace short_window calculation: long_window // 2 -> int(long_window * frac)."""
    # Match: short_window = long_window // N  OR  short_window = int(long_window * N)
    pattern = r'(short_window\s*=\s*)(long_window\s*//\s*\d+|int\(long_window\s*\*\s*[\d.]+\))'
    replacement = rf'\g<1>int(long_window * {value:.10g})'
    new_content, count = re.subn(pattern, replacement, content)
    if count == 0:
        raise ValueError("Could not find short_window calculation in train.py")
    return new_content


def _replace_lm_head_wd(content: str, value: float) -> str:
    """Replace weight_decay in the lm_head optimizer config."""
    pattern = (r'("lm_head" in path[\s\S]*?"weight_decay":\s*)([\d.]+)')
    new_content, count = re.subn(pattern, rf'\g<1>{value:.10g}', content, count=1)
    if count == 0:
        raise ValueError("Could not find lm_head weight_decay in train.py")
    return new_content


def _replace_mlp_expansion(content: str, value: float) -> str:
    """Replace MLP expansion factor in the 'hidden = int(N * config.n_embd)' line."""
    val_str = f"{value:.10g}" if isinstance(value, float) else str(value)

    # Match: hidden = int(N * config.n_embd)
    pattern = r'(hidden\s*=\s*int\()(\d+(?:\.\d+)?)\s*\*\s*config\.n_embd(\))'

    # Find the MLP class and replace within it
    mlp_pattern = r'(class MLP[\s\S]*?def __call__)'
    mlp_match = re.search(mlp_pattern, content)
    if not mlp_match:
        raise ValueError("Could not find MLP class in train.py")

    mlp_section = mlp_match.group(0)
    new_mlp = re.sub(pattern, rf'\g<1>{val_str} * config.n_embd\3', mlp_section)
    new_content = content[:mlp_match.start()] + new_mlp + content[mlp_match.end():]
    if new_content == content:
        raise ValueError("Could not find MLP expansion factor (hidden = int(N * config.n_embd)) in train.py")
    return new_content


def _replace_activation(content: str, value) -> str:
    """Replace activation function in MLP.__call__."""
    if isinstance(value, (int, float)):
        mapping = CATEGORICAL_VALUES.get("ACTIVATION", {})
        act_name = mapping.get(int(value), "gelu")
    else:
        act_name = str(value)

    # Map activation names to MLX code
    act_code = {
        "gelu": "nn.gelu(x)",
        "relu_sq": "mx.maximum(x, 0) ** 2",
        "swiglu": "nn.silu(x)",  # simplified SwiGLU
    }
    code = act_code.get(act_name, f"nn.{act_name}(x)")

    # Replace the activation line in MLP.__call__
    # Match various activation patterns
    pattern = r'(        x = )(nn\.\w+\(x\)|mx\.square\(nn\.relu\(x\)\)|mx\.maximum\(x,\s*0\)\s*\*\*\s*2)'
    replacement = rf'\g<1>{code}'
    new_content, count = re.subn(pattern, replacement, content)
    if count == 0:
        raise ValueError("Could not find activation in MLP in train.py")
    return new_content


def apply_config(config: dict, train_py_path: str | Path) -> list[str]:
    """
    Apply a configuration dict to train.py by modifying hyperparameter constants.

    Args:
        config: Dict of factor_name -> value.
        train_py_path: Path to the train.py file to modify.

    Returns:
        List of changes made (for logging).
    """
    path = Path(train_py_path)
    content = path.read_text()
    changes = []

    for factor_name, value in config.items():
        const_name = FACTOR_TO_CONSTANT.get(factor_name)

        try:
            if factor_name == "TOTAL_BATCH_SIZE_EXP":
                content = _replace_total_batch_size(content, int(value))
                changes.append(f"TOTAL_BATCH_SIZE = 2**{int(value)}")

            elif factor_name == "ADAM_BETA1":
                beta2 = config.get("ADAM_BETA2")
                if beta2 is None:
                    beta2 = _read_adam_beta2(content)
                content = _replace_adam_betas(content, float(value), float(beta2))
                changes.append(f"ADAM_BETAS = ({value}, {beta2})")

            elif factor_name == "ADAM_BETA2":
                if "ADAM_BETA1" not in config:
                    beta1 = _read_adam_beta1(content)
                    content = _replace_adam_betas(content, float(beta1), float(value))
                    changes.append(f"ADAM_BETAS = ({beta1}, {value})")

            elif factor_name == "WINDOW_PATTERN":
                content = _replace_window_pattern(content, value)
                if isinstance(value, (int, float)):
                    mapping = CATEGORICAL_VALUES.get("WINDOW_PATTERN", {})
                    changes.append(f"WINDOW_PATTERN = \"{mapping.get(int(value), 'SSSL')}\"")
                else:
                    changes.append(f"WINDOW_PATTERN = \"{value}\"")

            elif factor_name == "USE_MUON":
                if isinstance(value, (int, float)):
                    bool_val = bool(int(value))
                else:
                    bool_val = bool(value)
                content = _replace_constant(content, "USE_MUON", bool_val)
                changes.append(f"USE_MUON = {bool_val}")

            # --- Code-swap factors ---
            elif factor_name == "ROPE_BASE":
                content = _replace_rope_base(content, float(value))
                changes.append(f"RoPE base = {value}")

            elif factor_name == "EMBED_WD":
                content = _replace_embed_wd(content, float(value))
                changes.append(f"embed weight_decay = {value}")

            elif factor_name == "VE_WD":
                content = _replace_ve_wd(content, float(value))
                changes.append(f"value_embeds weight_decay = {value}")

            elif factor_name == "X0_LAMBDA_INIT":
                content = _replace_x0_lambda_init(content, float(value))
                changes.append(f"x0_lambdas init = {value}")

            elif factor_name == "INIT_SCALE":
                content = _replace_init_scale(content, float(value))
                changes.append(f"init scale = {value}")

            elif factor_name == "LOGIT_CAP":
                content = _replace_logit_cap(content, float(value))
                changes.append(f"logit cap = {value}")

            elif factor_name == "VE_GATE_CHANNELS":
                content = _replace_ve_gate_channels(content, int(value))
                changes.append(f"ve_gate_channels = {value}")

            elif factor_name == "SHORT_WINDOW_FRAC":
                content = _replace_short_window_frac(content, float(value))
                changes.append(f"short_window_frac = {value}")

            elif factor_name == "LM_HEAD_WD":
                content = _replace_lm_head_wd(content, float(value))
                changes.append(f"lm_head weight_decay = {value}")

            elif factor_name == "MLP_EXPANSION":
                content = _replace_mlp_expansion(content, float(value))
                changes.append(f"MLP expansion = {value}")

            elif factor_name == "ACTIVATION":
                content = _replace_activation(content, value)
                if isinstance(value, (int, float)):
                    mapping = CATEGORICAL_VALUES.get("ACTIVATION", {})
                    changes.append(f"activation = {mapping.get(int(value), 'gelu')}")
                else:
                    changes.append(f"activation = {value}")

            elif const_name is not None:
                content = _replace_constant(content, const_name, value)
                changes.append(f"{const_name} = {value}")

        except ValueError as e:
            changes.append(f"SKIPPED {factor_name}: {e}")

    path.write_text(content)
    return changes


def _read_adam_beta1(content: str) -> float:
    """Extract current beta1 from ADAM_BETAS = (b1, b2)."""
    m = re.search(r'ADAM_BETAS\s*=\s*\(([^,]+),', content)
    return float(m.group(1)) if m else 0.85


def _read_adam_beta2(content: str) -> float:
    """Extract current beta2 from ADAM_BETAS = (b1, b2)."""
    m = re.search(r'ADAM_BETAS\s*=\s*\([^,]+,\s*([^)]+)\)', content)
    return float(m.group(1)) if m else 0.99


def read_current_config(train_py_path: str | Path) -> dict:
    """
    Parse current hyperparameter values from train.py.

    Returns a dict of factor_name -> current_value for all known factors.
    """
    path = Path(train_py_path)
    content = path.read_text()
    config = {}

    # Simple constants
    simple_constants = [
        "DEPTH", "EMBEDDING_LR", "WEIGHT_DECAY", "MATRIX_LR",
        "WARMDOWN_RATIO", "UNEMBEDDING_LR", "DEVICE_BATCH_SIZE",
        "FINAL_LR_FRAC", "WARMUP_RATIO", "SCALAR_LR",
        "HEAD_DIM", "ASPECT_RATIO", "NS_STEPS", "MUON_MOMENTUM", "MUON_BETA2",
    ]
    for name in simple_constants:
        m = re.search(rf'^{re.escape(name)}\s*=\s*(.+)$', content, re.MULTILINE)
        if m:
            val_str = m.group(1).strip()
            try:
                config[name] = eval(val_str)  # Safe for numeric literals
            except Exception:
                config[name] = val_str

    # TOTAL_BATCH_SIZE
    m = re.search(r'TOTAL_BATCH_SIZE\s*=\s*2\*\*(\d+)', content)
    if m:
        config["TOTAL_BATCH_SIZE_EXP"] = int(m.group(1))

    # ADAM_BETAS
    config["ADAM_BETA1"] = _read_adam_beta1(content)
    config["ADAM_BETA2"] = _read_adam_beta2(content)

    # WINDOW_PATTERN
    m = re.search(r'WINDOW_PATTERN\s*=\s*"([^"]+)"', content)
    if m:
        config["WINDOW_PATTERN"] = m.group(1)

    # USE_MUON
    m = re.search(r'USE_MUON\s*=\s*(True|False)', content)
    if m:
        config["USE_MUON"] = m.group(1) == "True"

    # Code-swap factors
    m = re.search(r'nn\.RoPE\([^)]*base=\s*([\d.]+)', content)
    if m:
        config["ROPE_BASE"] = float(m.group(1))

    m = re.search(r'"wte" in path[\s\S]*?"weight_decay":\s*([\d.]+)', content)
    if m:
        config["EMBED_WD"] = float(m.group(1))

    m = re.search(r'"value_embeds" in path[\s\S]*?"weight_decay":\s*([\d.]+)', content)
    if m:
        config["VE_WD"] = float(m.group(1))

    m = re.search(r'mx\.full\(\(self\.config\.n_layer,\),\s*([\d.]+)', content)
    if m:
        config["X0_LAMBDA_INIT"] = float(m.group(1))

    m = re.search(r'scale\s*=\s*([\d.]+)\*\*0\.5\s*\*\s*n_embd\*\*-0\.5', content)
    if m:
        config["INIT_SCALE"] = float(m.group(1))

    m = re.search(r'logits\s*=\s*([\d.]+)\s*\*\s*mx\.tanh', content)
    if m:
        config["LOGIT_CAP"] = float(m.group(1))

    m = re.search(r'self\.ve_gate_channels\s*=\s*(\d+)', content)
    if m:
        config["VE_GATE_CHANNELS"] = int(m.group(1))

    m = re.search(r'short_window\s*=\s*long_window\s*//\s*(\d+)', content)
    if m:
        config["SHORT_WINDOW_FRAC"] = 1.0 / int(m.group(1))
    else:
        m = re.search(r'short_window\s*=\s*int\(long_window\s*\*\s*([\d.]+)\)', content)
        if m:
            config["SHORT_WINDOW_FRAC"] = float(m.group(1))

    m = re.search(r'"lm_head" in path[\s\S]*?"weight_decay":\s*([\d.]+)', content)
    if m:
        config["LM_HEAD_WD"] = float(m.group(1))

    # MLP expansion: look for int(N * config.n_embd) in MLP class
    mlp_match = re.search(r'class MLP[\s\S]*?def __call__', content)
    if mlp_match:
        mlp_section = mlp_match.group(0)
        m = re.search(r'int\(([\d.]+)\s*\*\s*config\.n_embd\)', mlp_section)
        if m:
            config["MLP_EXPANSION"] = float(m.group(1))

    # Activation — match various relu_sq and gelu patterns
    m = re.search(r'x = (nn\.\w+\(x\)|mx\.square\(nn\.relu\(x\)\)|mx\.maximum\(x,\s*0\)\s*\*\*\s*2)', content)
    if m:
        act_code = m.group(1)
        if "gelu" in act_code:
            config["ACTIVATION"] = "gelu"
        elif ("square" in act_code and "relu" in act_code) or ("maximum" in act_code and "** 2" in act_code):
            config["ACTIVATION"] = "relu_sq"
        elif "silu" in act_code:
            config["ACTIVATION"] = "swiglu"

    return config
