# Factorial Screening: Epoch Boundary Prompt

This file contains the prompt template used when `--llm` is passed to the orchestrator.
At each epoch boundary, the accumulated knowledge is formatted into this prompt
and sent to a language model to propose the next set of screening factors.

## Context

The system runs Plackett-Burman (Resolution III) fractional factorial screening
of LLM training hyperparameters on Apple Silicon (MLX). Each epoch screens
15-19 factors across K generations, then performs a factor reset. Factors that
have been reliably identified as significant are locked at their best values
and excluded from future screening.

## How Factors Are Chosen

1. **Locked factors** (confidence = "locked" or "high"): These have been tested
   multiple times with consistent, significant effects. They are fixed at their
   best value and should NOT be re-screened.

2. **Medium-confidence factors**: Tested once or twice with some significance.
   These should be re-screened at narrower ranges around the best known value.

3. **New factors**: Untested ideas from the rotation schedule or novel
   architectural/optimizer changes.

## What Makes a Good Factor Proposal

- **Actionable**: The factor must correspond to a modifiable constant or code
  pattern in train.py.
- **Two-level**: Each factor needs a low and high level that represent
  meaningfully different choices.
- **Independent**: Factors should be roughly independent. Use the dependency
  system for conditional factors (e.g., MUON_MOMENTUM depends on USE_MUON=True).
- **Impact-oriented**: Every factor should plausibly improve val_bpb.

## Knowledge JSON Structure

```json
{
  "global_best_val_bpb": 1.25,
  "global_best_config": {"DEPTH": 4, "EMBEDDING_LR": 0.8, ...},
  "locked_factors": {"DEPTH": 4, "TOTAL_BATCH_SIZE_EXP": 15},
  "factor_history": {
    "DEPTH": {
      "tested_epochs": [0],
      "effect_sizes": [-0.42],
      "significant_count": 1,
      "total_tests": 1
    }
  },
  "epochs": [...]
}
```
