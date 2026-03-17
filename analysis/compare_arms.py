"""
Compare K=3 vs K=4 arm performance from results TSVs.

Usage:
    uv run analysis/compare_arms.py
    uv run analysis/compare_arms.py --results-dir results/
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def load_tsv(path: Path) -> list[dict]:
    """Load a results TSV into a list of dicts."""
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def analyze_arm(rows: list[dict], arm_name: str) -> dict:
    """Compute summary statistics for one arm."""
    if not rows:
        return {"arm": arm_name, "total_runs": 0}

    val_bpbs = []
    crashes = 0
    timeouts = 0
    epochs = set()
    generations = set()

    for row in rows:
        try:
            bpb = float(row.get("val_bpb", "inf"))
            if bpb < float("inf"):
                val_bpbs.append(bpb)
            else:
                crashes += 1
        except (ValueError, TypeError):
            crashes += 1

        status = row.get("status", "")
        if status == "timeout":
            timeouts += 1
        epochs.add(row.get("epoch", "?"))
        generations.add((row.get("epoch", "?"), row.get("gen", "?")))

    return {
        "arm": arm_name,
        "total_runs": len(rows),
        "valid_runs": len(val_bpbs),
        "crashes": crashes,
        "timeouts": timeouts,
        "epochs_completed": len(epochs),
        "generations_run": len(generations),
        "best_val_bpb": min(val_bpbs) if val_bpbs else float("inf"),
        "worst_val_bpb": max(val_bpbs) if val_bpbs else float("inf"),
        "median_val_bpb": sorted(val_bpbs)[len(val_bpbs) // 2] if val_bpbs else float("inf"),
        "mean_val_bpb": sum(val_bpbs) / len(val_bpbs) if val_bpbs else float("inf"),
    }


def print_comparison(stats_a: dict, stats_b: dict, knowledge_dir: Path):
    """Print a formatted comparison report."""
    print("=" * 70)
    print("AUTORESEARCH-FACTORIAL: ARM COMPARISON REPORT")
    print("=" * 70)
    print()

    headers = ["Metric", "Arm A (K=3)", "Arm B (K=4)"]
    rows = [
        ("Total runs", stats_a.get("total_runs", 0), stats_b.get("total_runs", 0)),
        ("Valid runs", stats_a.get("valid_runs", 0), stats_b.get("valid_runs", 0)),
        ("Crashes", stats_a.get("crashes", 0), stats_b.get("crashes", 0)),
        ("Timeouts", stats_a.get("timeouts", 0), stats_b.get("timeouts", 0)),
        ("Epochs completed", stats_a.get("epochs_completed", 0), stats_b.get("epochs_completed", 0)),
        ("Generations run", stats_a.get("generations_run", 0), stats_b.get("generations_run", 0)),
    ]

    # Print table
    col_widths = [25, 20, 20]
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))
    for label, val_a, val_b in rows:
        print(f"  {label:<23}  {str(val_a):>18}  {str(val_b):>18}")

    print()
    print("--- Performance ---")
    metrics = [
        ("Best val_bpb", "best_val_bpb"),
        ("Median val_bpb", "median_val_bpb"),
        ("Mean val_bpb", "mean_val_bpb"),
        ("Worst val_bpb", "worst_val_bpb"),
    ]
    for label, key in metrics:
        va = stats_a.get(key, float("inf"))
        vb = stats_b.get(key, float("inf"))
        va_str = f"{va:.6f}" if va < float("inf") else "N/A"
        vb_str = f"{vb:.6f}" if vb < float("inf") else "N/A"
        marker = ""
        if va < float("inf") and vb < float("inf"):
            if va < vb:
                marker = " <-- WINNER"
            elif vb < va:
                marker = "                         <-- WINNER"
        print(f"  {label:<23}  {va_str:>18}  {vb_str:>18}{marker}")

    # Determine overall winner
    best_a = stats_a.get("best_val_bpb", float("inf"))
    best_b = stats_b.get("best_val_bpb", float("inf"))
    print()
    if best_a < best_b:
        print(f"WINNER: Arm A (K=3) with val_bpb = {best_a:.6f}")
    elif best_b < best_a:
        print(f"WINNER: Arm B (K=4) with val_bpb = {best_b:.6f}")
    else:
        print("TIE: Both arms have equal best val_bpb")

    # Show knowledge summary if available
    for arm_name in ["arm_a", "arm_b"]:
        k_path = knowledge_dir / f"{arm_name}_knowledge.json"
        if k_path.exists():
            k = json.loads(k_path.read_text())
            locked = k.get("locked_factors", {})
            if locked:
                print(f"\n{arm_name} locked factors: {json.dumps(locked, indent=2)}")

    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Compare factorial arm results")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory containing arm TSVs and knowledge files")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rows_a = load_tsv(results_dir / "arm_a.tsv")
    rows_b = load_tsv(results_dir / "arm_b.tsv")

    if not rows_a and not rows_b:
        print("No results found. Run the orchestrator first:")
        print("  uv run -m factorial.orchestrator")
        sys.exit(1)

    stats_a = analyze_arm(rows_a, "arm_a")
    stats_b = analyze_arm(rows_b, "arm_b")
    print_comparison(stats_a, stats_b, results_dir)


if __name__ == "__main__":
    main()
