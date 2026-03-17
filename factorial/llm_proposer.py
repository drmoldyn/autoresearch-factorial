"""
Optional LLM-based factor proposal at epoch boundaries.

When --llm is passed, the orchestrator calls this module to analyze accumulated
knowledge and propose new factors for the next screening epoch. Falls back to
the pre-defined rotation schedule if the LLM call fails.

Supports:
- Claude Code CLI (default)
- Any OpenAI-compatible endpoint (--llm-endpoint)
"""

import json
import subprocess
from pathlib import Path

from .factors import Factor


EPOCH_BOUNDARY_PROMPT = """\
You are the research director for an autonomous factorial screening system
optimizing LLM training hyperparameters on Apple Silicon (MLX).

## Current Knowledge
```json
{knowledge}
```

## Your Task
Propose 15-19 factors for the next screening epoch. For each factor provide
a JSON array of objects with these fields:
- "name": factor name (e.g. "DEPTH")
- "low": low level (number)
- "high": high level (number)
- "baseline": center/default value (number)
- "dtype": "int", "float", or "categorical"
- "rationale": one sentence explaining why to test this factor

## Rules
1. Factors with confidence="locked" in knowledge are LOCKED. Do NOT re-test.
2. Factors with confidence="high": consider locked, do not re-test.
3. Factors with confidence="medium": re-test at REFINED ranges (narrower around
   the best known value).
4. Include 4-8 UNTESTED factors from novel architectural or optimizer ideas.
5. Prioritize factors that may interact with known high-impact factors.
6. Every factor must serve the goal: lower val_bpb at the next checkpoint.

## Output Format
Return ONLY a JSON array. No markdown, no explanation. Example:
[
  {{"name": "DEPTH", "low": 3, "high": 5, "baseline": 4, "dtype": "int",
    "rationale": "Re-screen at narrower range around known optimum"}}
]
"""


def propose_factors_via_llm(
    knowledge_path: str | Path,
    endpoint: str | None = None,
) -> list[Factor]:
    """
    Call LLM to propose next epoch's factors based on accumulated knowledge.

    Args:
        knowledge_path: Path to knowledge.json.
        endpoint: Optional OpenAI-compatible API endpoint URL.

    Returns:
        List of Factor objects proposed by the LLM.
    """
    knowledge_path = Path(knowledge_path)
    if not knowledge_path.exists():
        return []

    knowledge = json.loads(knowledge_path.read_text())
    prompt = EPOCH_BOUNDARY_PROMPT.format(knowledge=json.dumps(knowledge, indent=2))

    if endpoint:
        response_text = _call_openai_compatible(endpoint, prompt)
    else:
        response_text = _call_claude_code(prompt)

    if not response_text:
        return []

    return _parse_factor_proposals(response_text)


def _call_claude_code(prompt: str) -> str:
    """Call Claude Code CLI to get factor proposals."""
    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "json"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            return ""
        # Claude Code JSON output has a "result" field
        try:
            output = json.loads(result.stdout)
            return output.get("result", result.stdout)
        except json.JSONDecodeError:
            return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def _call_openai_compatible(endpoint: str, prompt: str) -> str:
    """Call an OpenAI-compatible API endpoint."""
    try:
        import requests
        response = requests.post(
            f"{endpoint}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 4096,
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception:
        return ""


def _parse_factor_proposals(text: str) -> list[Factor]:
    """Parse LLM response into Factor objects."""
    # Extract JSON array from response
    text = text.strip()
    # Handle markdown code blocks
    if "```" in text:
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            text = text[start:end]

    try:
        proposals = json.loads(text)
    except json.JSONDecodeError:
        return []

    if not isinstance(proposals, list):
        return []

    factors = []
    for p in proposals:
        try:
            factors.append(Factor(
                name=p["name"],
                low=float(p["low"]),
                high=float(p["high"]),
                baseline=float(p["baseline"]),
                dtype=p.get("dtype", "float"),
            ))
        except (KeyError, ValueError, TypeError):
            continue

    return factors
