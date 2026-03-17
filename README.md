# autoresearch-factorial

Factorial-evolutionary autonomous ML research on Apple Silicon.

Replaces greedy one-factor-at-a-time hyperparameter search with **Plackett-Burman fractional factorial screening** — screening 19 factors simultaneously in 20 runs instead of testing them one by one. Factors are evolved across generations: significant factors are locked, new factors rotate in, and the system builds on accumulated knowledge at every step.

Built for [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) ecosystem. Runs on MLX, designed for M4 Max (128GB) and tested on M3 Ultra (512GB).

## Why Factorial Screening?

Standard autoresearch uses greedy hill climbing: change one hyperparameter, see if it helps, repeat. This has three problems:

1. **Misses interactions** — EMBEDDING_LR × WEIGHT_DECAY can flip signs depending on the other's value. One-at-a-time search can't detect this.
2. **Wastes runs** — Most hyperparameters have negligible effect. Greedy search spends one full training run to learn "this didn't matter."
3. **Local optima** — Moving one factor at a time gets trapped. Factorial designs explore the joint space.

Plackett-Burman designs screen N-1 factors in N runs (next multiple of 4). A 20-run design screens 19 factors simultaneously — each run provides information about *every* factor.

## How It Works

```
ORCHESTRATOR runs autonomously overnight
├── ARM A (K=3 generations per epoch)
│   └── Epoch 0: G0 → G1 → G2 → FACTOR RESET → Epoch 1: ...
├── ARM B (K=4 generations per epoch)
│   └── Epoch 0: G0 → G1 → G2 → G3 → FACTOR RESET → Epoch 1: ...
│
├── Hour 4: Compare arms → kill loser → both workers on winning K
└── Continuous: knowledge.json accumulates everything learned
```

### Each Generation

1. **Design**: Generate a Plackett-Burman matrix for current factors
2. **Run**: Execute each row of the matrix (apply config → train 5 min → record val_bpb)
3. **Analyze**: Compute main effects via contrast analysis, estimate SE via Lenth's method
4. **Foldover** (adaptive): Only where aliased effects are ambiguous (max 4 extra runs)
5. **Winner**: Construct best config from significant factors' best levels
6. **Refine**: Narrow factor ranges around winner for next generation

### Epoch Boundaries

After K generations, the system resets its factor set:
- **Lock** high-confidence factors at their best values (don't re-test)
- **Rotate in** new untested factors from the schedule
- **Refine** medium-confidence factors at narrower ranges
- Optionally **call an LLM** to propose novel factors based on accumulated knowledge

### Meta-Variable Testing

The number of generations before factor reset (K) is itself a meta-variable. The system tests K=3 vs K=4 in parallel for the first 4 hours, then converges on whichever produced better val_bpb.

## Quickstart

```bash
# Clone and set up
git clone https://github.com/YOUR_USERNAME/autoresearch-factorial.git
cd autoresearch-factorial

# Install dependencies
uv sync

# Prepare data (one-time, ~2 min)
uv run prepare.py

# Run the full autonomous system (overnight)
uv run -m factorial.orchestrator

# Or with LLM-based factor proposals at epoch boundaries
uv run -m factorial.orchestrator --llm

# Single-arm mode (for smaller machines or testing)
uv run -m factorial.orchestrator --single 3
```

## Configuration

The starting baseline is from the best published MLX result (PR #4, val_bpb ≈ 1.28):

| Parameter | Value |
|-----------|-------|
| DEPTH | 4 |
| TOTAL_BATCH_SIZE | 2^14 (16384) |
| EMBEDDING_LR | 0.8 |
| WEIGHT_DECAY | 0.05 |
| MATRIX_LR | 0.04 |
| USE_MUON | True |
| NS_STEPS | 5 |

### Epoch 0 Factors (19 factors, 20-run PB design)

**High-impact**: DEPTH, TOTAL_BATCH_SIZE, EMBEDDING_LR, WEIGHT_DECAY, MATRIX_LR

**Medium-impact**: WARMDOWN_RATIO, ROPE_BASE, EMBED_WD, VE_WD, X0_LAMBDA_INIT, INIT_SCALE

**Exploratory**: UNEMBEDDING_LR, ADAM_BETA1, ADAM_BETA2, WINDOW_PATTERN, DEVICE_BATCH_SIZE, FINAL_LR_FRAC, WARMUP_RATIO, SCALAR_LR

### Rotation Factors (Epoch 1+)

USE_MUON, NS_STEPS, MUON_MOMENTUM, MUON_BETA2, LOGIT_CAP, VE_GATE_CHANNELS, HEAD_DIM, ASPECT_RATIO, SHORT_WINDOW_FRAC, LM_HEAD_WD, MLP_EXPANSION, ACTIVATION

## Project Structure

```
autoresearch-factorial/
├── train.py                 # MLX training script (modified by orchestrator)
├── prepare.py               # Data prep (immutable, from autoresearch-mlx)
├── program.md               # LLM prompt template for epoch boundaries
├── factorial/
│   ├── orchestrator.py      # Autonomous overnight runner
│   ├── designer.py          # PB matrix generation + foldovers
│   ├── analyzer.py          # Effect computation, Lenth's SE, winner selection
│   ├── applicator.py        # Config → train.py writer
│   ├── factors.py           # Factor definitions, constraints, rotation schedule
│   ├── knowledge.py         # Persistent JSON knowledge store
│   └── llm_proposer.py      # Optional LLM-based factor proposals
├── analysis/
│   └── compare_arms.py      # K=3 vs K=4 comparison report
└── results/                 # Auto-generated during runs
    ├── arm_a.tsv            # Per-run results
    ├── arm_b.tsv
    ├── effects/             # Per-generation effect analysis
    └── *_knowledge.json     # Accumulated knowledge
```

## Overnight Run Budget

| Phase | Duration | Experiments | Notes |
|-------|----------|-------------|-------|
| Parallel burn-in | 0-4h | ~34 | Both arms, shared GPU (~14 min/run) |
| Convergence | 4h+ | ~50 | Both workers on winning K |
| **Total (10h)** | | **~84** | Full factorial analysis throughout |

Compare: vanilla autoresearch runs ~85 greedy experiments in 10h but with no effect analysis, no interaction detection, no systematic factor ranking.

## Key Design Decisions

- **Resolution III**: Sufficient for screening. Main effects are aliased with 2-factor interactions, but adaptive foldovers de-alias only where it matters.
- **Lenth's method**: Estimates standard error from the median of absolute effects — no replicate runs needed.
- **Factor locking**: High-confidence factors are locked after 2+ consistent significant tests. This focuses compute on what's still unknown.
- **Crash tolerance**: Failed runs get val_bpb = infinity. The system continues without stopping.
- **Checkpoint/resume**: State is saved after every generation. Survives crashes, sleep, and reboots.

## Hardware Requirements

- **Full mode** (2 workers): ~54GB unified memory → M3/M4 Ultra or higher
- **Single-arm mode**: ~27GB → M4 Max (128GB) with room to spare
- **Post-convergence**: Both workers fit on 128GB M4 Max (~27GB each)

## Comparing Results

```bash
# Generate comparison report
uv run analysis/compare_arms.py

# Check accumulated knowledge
cat results/arm_a_knowledge.json | python -m json.tool
```

## References

- [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — the original evolutionary ML research framework
- [autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) — Apple Silicon port (our base)
- Plackett, R.L. & Burman, J.P. (1946) — "The Design of Optimum Multifactorial Experiments"
- Lenth, R.V. (1989) — "Quick and Easy Analysis of Unreplicated Factorials"
