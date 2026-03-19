"""Reset knowledge store and checkpoint to knowledge-optimized baseline.

Synthesizes 963 experiments of accumulated knowledge:
- Starts from global best validation config (val_bpb=1.241339)
- Adds new factor discoveries (Z_LOSS_WEIGHT=0.0, etc.)
- Unlocks wrongly-locked factors (TOTAL_BATCH_SIZE_EXP, WINDOW_PATTERN)
- Locks Z_LOSS_WEIGHT=0.0 (overwhelming evidence: 48 tests, 33 sig, 100% consistent)
- Resets calibrating factor ranges to wider intervals around correct centers

Run: python reset_baseline.py
"""

import json
import shutil
from pathlib import Path

RESULTS_DIR = Path("results")
KNOWLEDGE_PATH = RESULTS_DIR / "arm_k4_knowledge.json"
CHECKPOINT_PATH = RESULTS_DIR / "arm_k4_checkpoint.json"

# ─── KNOWLEDGE-OPTIMIZED BASELINE ───────────────────────────────────────
# Foundation: global best validation config (1.241339)
# + new factors from current run knowledge
OPTIMIZED_BASELINE = {
    "ACTIVATION": 1,
    "ADAM_BETA1": 0.8,
    "ADAM_BETA2": 0.9,
    "ASPECT_RATIO": 64,
    "CAUTIOUS_WD": 1,
    "DEPTH": 4,
    "DEVICE_BATCH_SIZE": 8,
    "EMBEDDING_LR": 0.3,
    "EMBED_WD": 0.0,
    "FINAL_LR_FRAC": 0.0,
    "HEAD_DIM": 128,
    "INIT_SCALE": 4.0,
    "KV_HEAD_RATIO": 0,
    "LM_HEAD_WD": 0.0025,
    "LOGIT_CAP": 30,
    "MATRIX_LR": 0.05,
    "MLP_EXPANSION": 4.0,
    "MUON_BETA2": 0.97425,
    "MUON_MOMENTUM": 0.85,
    "NS_STEPS": 5,
    "RESID_LR_RATIO": 0.0258,
    "ROPE_BASE": 10000,
    "SCALAR_LR": 0.3,
    "SHORT_WINDOW_FRAC": 0.125,
    "SPARSE_ATTN_GATE": 0,
    "TOTAL_BATCH_SIZE_EXP": 15,
    "UNEMBEDDING_LR": 0.004,
    "USE_MUON": 1,
    "VE_GATE_CHANNELS": 52,
    "VE_WD": 0.0,
    "WARMDOWN_RATIO": 0.5,
    "WARMUP_RATIO": 0.0,
    "WEIGHT_DECAY": 0.2,
    "WINDOW_PATTERN": 0,
    "X0_LAMBDA_INIT": 0.15,
    "Z_LOSS_WEIGHT": 0.0,
}

# ─── LOCK CHANGES ───────────────────────────────────────────────────────
# Factors that should be locked (all match global best)
CORRECT_LOCKS = {
    "USE_MUON": 1,
    "DEPTH": 4,
    "DEVICE_BATCH_SIZE": 8,
    "ASPECT_RATIO": 64,
    "ROPE_BASE": 10000,
    "VE_WD": 0.0,
    "MATRIX_LR": 0.05,
    "X0_LAMBDA_INIT": 0.15,
    "FINAL_LR_FRAC": 0.0,
    "WARMUP_RATIO": 0.0,
    "SPARSE_ATTN_GATE": 0,
    # NEW LOCK: overwhelming evidence (48 tests, 33 sig, 100% direction)
    "Z_LOSS_WEIGHT": 0.0,
}
# REMOVED from locks: TOTAL_BATCH_SIZE_EXP (was 14, global best=15)
# REMOVED from locks: WINDOW_PATTERN (was 1, global best=0)

# ─── CALIBRATING FACTOR RANGES ──────────────────────────────────────────
# Reset to wider ranges centered on global best values
RESET_CALIBRATIONS = {
    "ADAM_BETA1": {"best_value": 0.8, "range_low": 0.7, "range_high": 0.9},
    "EMBED_WD": {"best_value": 0.0, "range_low": 0.0, "range_high": 0.005},
    "HEAD_DIM": {"best_value": 128, "range_low": 96, "range_high": 160},
    "INIT_SCALE": {"best_value": 4.0, "range_low": 2.0, "range_high": 6.0},
    "LM_HEAD_WD": {"best_value": 0.0025, "range_low": 0.0, "range_high": 0.01},
    "MUON_MOMENTUM": {"best_value": 0.85, "range_low": 0.80, "range_high": 0.95},
    "SHORT_WINDOW_FRAC": {"best_value": 0.125, "range_low": 0.0625, "range_high": 0.25},
    "UNEMBEDDING_LR": {"best_value": 0.004, "range_low": 0.001, "range_high": 0.008},
    "VE_GATE_CHANNELS": {"best_value": 52, "range_low": 16, "range_high": 80},
    "WEIGHT_DECAY": {"best_value": 0.2, "range_low": 0.05, "range_high": 0.4},
}
# Z_LOSS_WEIGHT removed from calibrating → locked instead


def reset_knowledge(path: Path) -> None:
    """Reset knowledge store with corrected locks, calibrations, and baseline."""
    data = json.loads(path.read_text())

    print(f"Knowledge store loaded: {len(data.get('epochs', []))} epochs, "
          f"{len(data.get('validations', []))} validations")

    # 1. Fix locked factors
    old_locks = data.get("locked_factors", {})
    print(f"\nOld locks: {list(old_locks.keys())}")
    data["locked_factors"] = dict(CORRECT_LOCKS)
    print(f"New locks: {list(CORRECT_LOCKS.keys())}")
    unlocked = set(old_locks.keys()) - set(CORRECT_LOCKS.keys())
    new_locked = set(CORRECT_LOCKS.keys()) - set(old_locks.keys())
    if unlocked:
        print(f"  UNLOCKED: {unlocked}")
    if new_locked:
        print(f"  NEWLY LOCKED: {new_locked}")

    # 2. Fix calibrating factors
    old_cal = data.get("calibrating_factors", {})
    print(f"\nOld calibrating: {list(old_cal.keys())}")

    new_cal = {}
    for name, reset in RESET_CALIBRATIONS.items():
        # Preserve metadata from existing calibration if available
        existing = old_cal.get(name, {})
        new_cal[name] = {
            "best_value": reset["best_value"],
            "range_low": reset["range_low"],
            "range_high": reset["range_high"],
            # Preserve test counts and epoch data
            "n_tests": existing.get("n_tests", 0),
            "n_significant": existing.get("n_significant", 0),
            "epoch_entered": existing.get("epoch_entered", 0),
            "n_calibrations": 0,  # Reset calibration count
            "tested_epochs": existing.get("tested_epochs", []),
        }
        old_range = f"[{existing.get('range_low', '?')}, {existing.get('range_high', '?')}]"
        new_range = f"[{reset['range_low']}, {reset['range_high']}]"
        old_best = existing.get("best_value", "?")
        print(f"  {name}: {old_best} {old_range} -> {reset['best_value']} {new_range}")

    data["calibrating_factors"] = new_cal
    print(f"New calibrating: {list(new_cal.keys())}")

    # 3. Update baseline_config reference
    data["baseline_config"] = dict(OPTIMIZED_BASELINE)

    # 4. Keep factor_history intact (significance data is valuable)
    fh = data.get("factor_history", {})
    print(f"\nFactor history preserved: {len(fh)} factors")

    # 5. Keep validations intact (regression gate reference)
    vals = data.get("validations", [])
    print(f"Validations preserved: {len(vals)} records")
    print(f"Global best: {data.get('global_best_val_bpb', '?')}")

    # 6. Keep epochs data intact (screening evidence)
    print(f"Epoch data preserved: {len(data.get('epochs', []))} epochs")

    # Save
    path.write_text(json.dumps(data, indent=2, default=str))
    print(f"\nKnowledge store saved to {path}")


def reset_checkpoint(path: Path) -> None:
    """Reset checkpoint with optimized baseline and epoch=0."""
    data = json.loads(path.read_text())

    print(f"\nCheckpoint loaded: epoch={data.get('epoch')}, "
          f"gen={data.get('generation')}, "
          f"experiments={data.get('total_experiments')}")

    # Show key baseline differences
    old_baseline = data.get("current_baseline", {})
    changes = []
    for key in sorted(set(list(old_baseline.keys()) + list(OPTIMIZED_BASELINE.keys()))):
        old_val = old_baseline.get(key)
        new_val = OPTIMIZED_BASELINE.get(key)
        if old_val != new_val:
            changes.append(f"  {key}: {old_val} -> {new_val}")
    if changes:
        print(f"\nBaseline changes ({len(changes)} factors):")
        for c in changes:
            print(c)

    data["current_baseline"] = dict(OPTIMIZED_BASELINE)
    data["epoch"] = 0
    data["generation"] = 0
    # Keep total_experiments for record
    data["timestamp"] = __import__("time").strftime("%Y-%m-%d %H:%M:%S")

    path.write_text(json.dumps(data, indent=2, default=str))
    print(f"\nCheckpoint saved: epoch=0, gen=0, baseline=optimized")


def main():
    print("=" * 60)
    print("BASELINE RESET: Knowledge-Optimized Restart")
    print("=" * 60)

    # Backup
    for p in [KNOWLEDGE_PATH, CHECKPOINT_PATH]:
        bak = p.with_suffix(".json.bak")
        shutil.copy2(p, bak)
        print(f"Backup: {p} -> {bak}")

    reset_knowledge(KNOWLEDGE_PATH)
    reset_checkpoint(CHECKPOINT_PATH)

    print("\n" + "=" * 60)
    print("RESET COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Verify: python -c \"import json; d=json.load(open('results/arm_k4_checkpoint.json')); print(d['current_baseline'])\"")
    print("  2. Restart: uv run -m factorial.orchestrator --single 4 --max-parallel 4")


if __name__ == "__main__":
    main()
