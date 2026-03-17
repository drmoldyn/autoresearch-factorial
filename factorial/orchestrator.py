"""
Autonomous overnight orchestrator for factorial-evolutionary ML research.

Launches two parallel arms (K=3 and K=4 generations per epoch), compares
after 4 hours, then consolidates on the winning strategy. Each arm runs
Plackett-Burman screening generations with adaptive foldovers, accumulating
knowledge across epochs.

Usage:
    uv run -m factorial.orchestrator              # default (rotation mode)
    uv run -m factorial.orchestrator --llm        # use LLM at epoch boundaries
    uv run -m factorial.orchestrator --single 3   # single arm, K=3
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .analyzer import (
    adaptive_locking_threshold,
    adaptive_screening_threshold,
    compute_main_effects,
    compute_standard_error_lenth,
    decide_foldovers,
    generation_summary,
    rank_factors,
    recompute_with_foldover,
    select_winner,
)
from .applicator import apply_config, read_current_config
from .designer import (
    design_to_configs,
    generate_foldover,
    generate_pb_design,
    get_alias_structure,
)
from .factors import (
    EPOCH_0_FACTORS,
    Factor,
    check_constraints,
    check_dependencies,
    clear_muon_dependent_locks,
    fix_config,
    get_factor_rotation,
)
from .knowledge import KnowledgeStore
from .strategy import GenerationStrategy

PROJECT_ROOT = Path(__file__).parent.parent
CONVERGENCE_HOURS = 4.0
RUN_TIMEOUT = 900  # 15 min max per experiment


class ArmWorker:
    """Runs one arm (K=3 or K=4) in its own workspace."""

    def __init__(
        self,
        arm_k: int,
        arm_name: str,
        baseline_config: dict,
        workspace_dir: Path,
        results_dir: Path,
        llm_mode: str = "rotation",
        max_parallel: int = 0,
    ):
        self.arm_k = arm_k
        self.arm_name = arm_name
        self.workspace = Path(workspace_dir)
        self.results_dir = Path(results_dir)
        self.results_tsv = self.results_dir / f"{arm_name}.tsv"
        self.knowledge = KnowledgeStore(self.results_dir / f"{arm_name}_knowledge.json")
        self.checkpoint_path = self.results_dir / f"{arm_name}_checkpoint.json"
        self.current_baseline = dict(baseline_config)
        self.epoch = 0
        self.generation = 0
        self.llm_mode = llm_mode
        self.train_py = self.workspace / "train.py"
        self.total_experiments = 0
        self.start_time = time.time()
        self.llm_proposed_factors: list[Factor] | None = None
        self.max_parallel = max_parallel if max_parallel > 0 else self._detect_parallel_budget()
        self._recent_stderr: list[str] = []  # Collected by circuit breaker

        # Ensure results TSV exists with header
        if not self.results_tsv.exists():
            self.results_dir.mkdir(parents=True, exist_ok=True)
            with open(self.results_tsv, "w", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow([
                    "timestamp", "arm", "epoch", "gen", "run_label",
                    "val_bpb", "peak_vram_mb", "status", "config_summary",
                ])

    def log(self, msg: str):
        elapsed = (time.time() - self.start_time) / 3600
        print(f"[{self.arm_name} {elapsed:.1f}h] {msg}", flush=True)

    def run_single_experiment(self, config: dict, run_label: str) -> float:
        """Apply config to train.py, run training, return val_bpb."""
        # Apply config
        all_factors = EPOCH_0_FACTORS + list(get_factor_rotation(self.epoch, set(), self.knowledge))
        fixed_config, fixes = fix_config(all_factors, config)
        if fixes:
            self.log(f"  Config fixes: {fixes}")

        changes = apply_config(fixed_config, self.train_py)
        config_summary = "; ".join(changes[:5])  # Truncate for TSV

        # Run train.py
        self.log(f"  Running {run_label}...")
        try:
            result = subprocess.run(
                ["uv", "run", str(self.train_py)],
                capture_output=True, text=True, timeout=RUN_TIMEOUT,
                cwd=str(self.workspace),
                env={**os.environ, "PYTHONPATH": str(self.workspace)},
            )
            val_bpb = self._parse_val_bpb(result.stdout)
            peak_vram = self._parse_peak_vram(result.stdout)
            status = "ok"

            if val_bpb is None or (val_bpb is not None and not (val_bpb == val_bpb)):
                # val_bpb is None (not found) or NaN (training diverged)
                was_nan = val_bpb is not None
                val_bpb = float("inf")
                status = "nan" if was_nan else "crash"
                self.log(f"  {'NAN' if was_nan else 'CRASH'}: {'training diverged' if was_nan else 'no val_bpb in output'}")
                if not was_nan:
                    stderr_tail = result.stderr[-500:] if result.stderr else ""
                    stdout_tail = result.stdout[-500:] if result.stdout else ""
                    self.log(f"  stderr: {stderr_tail}")
                    self.log(f"  stdout: {stdout_tail}")
            else:
                self.log(f"  val_bpb: {val_bpb:.6f} | vram: {peak_vram:.0f}MB")

        except subprocess.TimeoutExpired:
            val_bpb = float("inf")
            peak_vram = 0
            status = "timeout"
            self.log(f"  TIMEOUT after {RUN_TIMEOUT}s")
        except Exception as e:
            val_bpb = float("inf")
            peak_vram = 0
            status = f"error:{e}"
            self.log(f"  ERROR: {e}")

        # Log to TSV
        with open(self.results_tsv, "a", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                self.arm_name, self.epoch, self.generation, run_label,
                f"{val_bpb:.6f}" if val_bpb != float("inf") else "inf",
                f"{peak_vram:.0f}" if peak_vram else "0",
                status, config_summary,
            ])

        self.total_experiments += 1
        return val_bpb

    def _parse_val_bpb(self, stdout: str) -> float | None:
        for line in stdout.split("\n"):
            if line.startswith("val_bpb:"):
                try:
                    return float(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass
        return None

    def _parse_peak_vram(self, stdout: str) -> float:
        for line in stdout.split("\n"):
            if line.startswith("peak_vram_mb:"):
                try:
                    return float(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass
        return 0.0

    @staticmethod
    def _detect_parallel_budget(n_arms: int = 1) -> int:
        """Detect max parallel experiments per arm based on system memory."""
        try:
            result = subprocess.run(
                ['sysctl', '-n', 'hw.memsize'],
                capture_output=True, text=True, timeout=5,
            )
            total_gb = int(result.stdout.strip()) / (1024 ** 3)
        except Exception:
            total_gb = 128
        # Reserve 40GB for OS. Divide remaining across arms. ~27GB per experiment.
        per_arm_gb = (total_gb - 40) / max(1, n_arms)
        max_parallel = max(1, int(per_arm_gb / 27))
        return min(max_parallel, 8)  # Cap at 8 to limit GPU contention

    def run_experiment_pool(
        self,
        configs: list[dict],
        labels: list[str],
    ) -> list[float]:
        """Run experiments using a worker pool. Freed slots are immediately reused.

        When a training run exits early (NaN, crash), the slot is immediately
        recycled for the next pending experiment. This avoids wasting time
        waiting for doomed runs to exhaust TIME_BUDGET.
        """
        n = len(configs)
        if n == 0:
            return []

        all_factors = EPOCH_0_FACTORS + list(
            get_factor_rotation(self.epoch, set(), self.knowledge)
        )

        results = [None] * n  # Indexed by config position
        queue = list(range(n))  # Indices into configs/labels still to launch
        # active: slot_id -> (proc, config_idx, label, config_summary, slot_path, launch_time)
        active: dict[int, tuple] = {}
        n_slots = min(self.max_parallel, n)

        self.log(f"  Pool: {n} experiments across {n_slots} slots")

        def _launch(slot_id: int, config_idx: int):
            """Prepare slot file, apply config, launch subprocess."""
            config = configs[config_idx]
            label = labels[config_idx]
            slot_path = self.workspace / f"train_slot{slot_id}.py"
            shutil.copy2(self.train_py, slot_path)

            fixed_config, fixes = fix_config(all_factors, config)
            if fixes:
                self.log(f"  Config fixes [{label}]: {fixes}")
            changes = apply_config(fixed_config, slot_path)
            config_summary = "; ".join(changes[:5])

            proc = subprocess.Popen(
                ["uv", "run", str(slot_path)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                cwd=str(self.workspace),
                env={**os.environ, "PYTHONPATH": str(self.workspace)},
            )
            active[slot_id] = (proc, config_idx, label, config_summary, slot_path, time.monotonic())

        def _collect(slot_id: int):
            """Collect result from a finished process in this slot."""
            proc, config_idx, label, config_summary, slot_path, _launch_t = active[slot_id]
            val_bpb, peak_vram, status = self._collect_experiment(proc, label)

            with open(self.results_tsv, "a", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    self.arm_name, self.epoch, self.generation, label,
                    f"{val_bpb:.6f}" if val_bpb != float("inf") else "inf",
                    f"{peak_vram:.0f}" if peak_vram else "0",
                    status, config_summary,
                ])

            self.total_experiments += 1
            results[config_idx] = val_bpb
            slot_path.unlink(missing_ok=True)
            del active[slot_id]

        # Fill initial slots
        for slot_id in range(n_slots):
            if queue:
                _launch(slot_id, queue.pop(0))

        self.log(f"  Launched {len(active)} experiments")

        # Poll loop: collect finished, enforce timeouts, launch pending
        while active:
            now = time.monotonic()
            finished_slots = []
            for slot_id, (proc, config_idx, label, cfg_sum, slot_path, launch_t) in list(active.items()):
                if proc.poll() is not None:
                    finished_slots.append(slot_id)
                elif now - launch_t > RUN_TIMEOUT:
                    # Wall-clock timeout: kill hung process
                    self.log(f"  [{label}] WALL-CLOCK TIMEOUT ({RUN_TIMEOUT}s) — killing")
                    proc.kill()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        pass  # Already killed, best effort
                    finished_slots.append(slot_id)

            for slot_id in finished_slots:
                _collect(slot_id)
                # Immediately reuse freed slot
                if queue:
                    _launch(slot_id, queue.pop(0))
                    self.log(f"  Slot {slot_id} recycled ({len(queue)} remaining)")

            if active and not finished_slots:
                time.sleep(0.5)  # Brief poll interval

        return results

    def _collect_experiment(
        self, proc: subprocess.Popen, label: str,
    ) -> tuple[float, float, str]:
        """Collect result from a finished experiment subprocess."""
        try:
            stdout, stderr = proc.communicate(timeout=10)  # Already done, just read
            val_bpb = self._parse_val_bpb(stdout)
            peak_vram = self._parse_peak_vram(stdout)
            status = "ok"

            if val_bpb is None or (val_bpb is not None and not (val_bpb == val_bpb)):
                was_nan = val_bpb is not None
                val_bpb = float("inf")
                status = "nan" if was_nan else "crash"
                self.log(f"  [{label}] {'NAN' if was_nan else 'CRASH'}")
                if not was_nan:
                    stderr_tail = (stderr or '')[-500:]
                    self.log(f"  [{label}] stderr: {stderr_tail[-300:]}")
                    self._recent_stderr.append(stderr_tail)
            else:
                self.log(f"  [{label}] val_bpb: {val_bpb:.6f} | vram: {peak_vram:.0f}MB")

        except subprocess.TimeoutExpired:
            proc.kill()
            _, stderr = proc.communicate()
            val_bpb = float("inf")
            peak_vram = 0
            status = "timeout"
            self.log(f"  [{label}] TIMEOUT")
            self._recent_stderr.append((stderr or '')[-500:])

        except Exception as e:
            val_bpb = float("inf")
            peak_vram = 0
            status = f"error:{e}"
            self.log(f"  [{label}] ERROR: {e}")
            self._recent_stderr.append(str(e))

        return val_bpb, peak_vram, status

    def _diagnose_crashes(self, crash_rate: float, n_crashes: int, n_total: int,
                          stderr_samples: list[str]) -> str:
        """Call LLM to diagnose generation crashes. Returns recommended action."""
        # Classify crash type from stderr
        oom_count = sum(1 for s in stderr_samples if "OutOfMemory" in s or "Insufficient Memory" in s)
        is_oom = oom_count > len(stderr_samples) * 0.5

        if is_oom:
            self.log(f"  CIRCUIT BREAKER: {n_crashes}/{n_total} OOM crashes ({crash_rate:.0%})")
            if self.max_parallel > 1:
                old = self.max_parallel
                self.max_parallel = max(1, self.max_parallel // 2)
                self.log(f"  Reducing parallelism: {old} → {self.max_parallel}")
            self.log(f"  Sleeping 60s for GPU memory recovery...")
            time.sleep(60)
            return "reduce_parallel"

        # For non-OOM crashes, call LLM for diagnosis
        diag_prompt = f"""CRASH TRIAGE ONLY. You are a crash classifier for an autonomous ML experiment runner.

Your ONLY job: classify the crash cause and recommend ONE action.
Do NOT suggest code changes, new factors, architectural modifications, or workflow changes.

Crash rate: {crash_rate:.0%} ({n_crashes}/{n_total} experiments crashed)
Epoch: {self.epoch}, Generation: {self.generation}

Sample stderr (up to 3 runs):
{'---'.join(stderr_samples[:3])}

In 1-2 sentences, classify the crash type (e.g. OOM, NaN divergence, shape mismatch, import error, timeout).
Then on the LAST LINE, output exactly one of these three words:
- continue — crashes are expected (extreme factor levels, NaN from aggressive LR)
- reduce_parallel — memory pressure causing crashes
- skip_generation — fundamental error affecting most/all runs"""

        try:
            from .llm_proposer import _call_claude_code
            response = _call_claude_code(diag_prompt)
            self.log(f"  LLM diagnosis: {response.strip()[:200]}")
            # Extract action from last line
            action = response.strip().split("\n")[-1].strip().lower()
            if action in ("continue", "reduce_parallel", "skip_generation"):
                return action
        except Exception as e:
            self.log(f"  LLM diagnosis failed: {e}")

        # Default: continue if <60% crash, skip if >=60%
        return "skip_generation" if crash_rate >= 0.6 else "continue"

    def run_generation(self, factors: list[Factor]) -> dict:
        """Run one PB generation: design -> execute -> analyze -> winner."""
        gen_start_time = time.monotonic()

        # Adaptive threshold based on accumulated knowledge
        pi = self.knowledge.get_active_fraction()
        screen_t = adaptive_screening_threshold(pi)
        self.log(f"=== Epoch {self.epoch}, Generation {self.generation} "
                 f"({len(factors)} factors, K={self.arm_k}) ===")
        self.log(f"  Adaptive threshold: π={pi:.2f} → t_screen={screen_t:.2f}")

        # Generate design matrix
        design = generate_pb_design(len(factors))
        factor_names = [f.name for f in factors]
        configs = design_to_configs(design, factors, self.current_baseline)
        n_runs = design.shape[0]

        self.log(f"  PB design: {n_runs} runs for {len(factors)} factors")

        # Run all experiments via worker pool (freed slots reused immediately)
        run_labels = [
            f"e{self.epoch}_g{self.generation}_run{j:02d}"
            for j in range(n_runs)
        ]
        responses = self.run_experiment_pool(configs, run_labels)

        # --- Circuit breaker: detect and diagnose crashes ---
        n_crashed = sum(1 for r in responses if r == float("inf"))
        if n_crashed > 0:
            crash_rate = n_crashed / len(responses)
            self.log(f"  Crash rate: {n_crashed}/{len(responses)} ({crash_rate:.0%})")
            action = self._diagnose_crashes(
                crash_rate, n_crashed, len(responses), self._recent_stderr,
            )
            if action == "skip_generation":
                self.log(f"  SKIPPING generation (crash rate too high)")
                self.generation += 1
                self.save_checkpoint()
                return dict(self.current_baseline)
        self._recent_stderr = []  # Reset after diagnosis

        responses_arr = np.array(responses)

        # Compute effects
        effects = compute_main_effects(design, responses_arr, factor_names)
        se = compute_standard_error_lenth(effects)
        ranked = rank_factors(effects, se, screening_threshold=screen_t)

        self.log(f"  SE (Lenth): {se:.6f}")
        self.log(f"  Significant factors:")
        for name, effect, t_ratio, sig in ranked:
            marker = "***" if sig else "   "
            direction = "better@high" if effect < 0 else "better@low"
            self.log(f"    {marker} {name}: effect={effect:.6f} t={t_ratio:.2f} ({direction})")

        # Adaptive foldover — accumulate all foldover data
        alias_struct = get_alias_structure(design, factor_names)
        foldover_names = decide_foldovers(
            effects, se, alias_struct, max_foldovers=4, screening_threshold=screen_t
        )
        foldover_responses = []
        # Track accumulated augmented design/responses across all foldovers
        augmented_design = design.copy()
        augmented_responses = responses_arr.copy()

        if foldover_names:
            self.log(f"  Running foldovers for: {foldover_names}")
            # Collect all foldover runs and batch them together
            all_fold_configs = []
            all_fold_labels = []
            all_fold_designs = []
            for fname in foldover_names:
                fidx = factor_names.index(fname)
                fold_design = generate_foldover(design, fidx)
                fold_configs = design_to_configs(fold_design, factors, self.current_baseline)
                all_fold_designs.append(fold_design)
                for j, fc in enumerate(fold_configs):
                    all_fold_configs.append(fc)
                    all_fold_labels.append(
                        f"e{self.epoch}_g{self.generation}_fold_{fname}_{j:02d}"
                    )

            # Run all foldover experiments via worker pool
            fold_resps_all = self.run_experiment_pool(all_fold_configs, all_fold_labels)
            foldover_responses.extend(fold_resps_all)

            # Accumulate all foldover designs into augmented matrix
            for fold_design in all_fold_designs:
                augmented_design = np.vstack([augmented_design, fold_design])
            augmented_responses = np.concatenate([
                responses_arr,
                np.array(fold_resps_all) if fold_resps_all else np.array([]),
            ])

            # Recompute effects using ALL accumulated data
            effects = compute_main_effects(augmented_design, augmented_responses, factor_names)
            se = compute_standard_error_lenth(effects)
            ranked = rank_factors(effects, se, screening_threshold=screen_t)
            self.log(f"  Post-foldover SE: {se:.6f}")

        # Select winner
        all_responses = np.array(responses + foldover_responses)
        winner = select_winner(
            configs, all_responses[:len(configs)], effects, se,
            self.current_baseline, screening_threshold=screen_t,
        )

        # Record to knowledge store
        summary = generation_summary(
            self.epoch, self.generation, self.arm_name,
            factors, effects, se, ranked, winner,
            all_responses, foldover_names,
        )
        self.knowledge.record_generation(self.epoch, self.generation, summary)

        # Save effects JSON
        effects_dir = self.results_dir / "effects"
        effects_dir.mkdir(parents=True, exist_ok=True)
        effects_path = effects_dir / f"{self.arm_name}_e{self.epoch}_g{self.generation}.json"
        effects_path.write_text(json.dumps(summary, indent=2, default=str))

        best_bpb = float(np.min(all_responses[np.isfinite(all_responses)])) \
            if np.any(np.isfinite(all_responses)) else float("inf")
        self.log(f"  Generation best val_bpb: {best_bpb:.6f}")
        self.log(f"  Winner config (significant factors): "
                 f"{summary.get('winner_config', {})}")

        # Update baseline with winner values for SIGNIFICANT factors only.
        # Non-significant factors stay at current baseline (don't adopt PB
        # extreme values that happened to be in the best run by chance).
        sig_factors = set(summary.get("significant_factors", []))
        for name, value in winner.items():
            if name in sig_factors or name in self.knowledge.get_locked_factors():
                self.current_baseline[name] = value

        gen_elapsed = time.monotonic() - gen_start_time

        # --- Per-generation metrics for retrospective K-optimization ---
        n_significant = len(sig_factors)
        gen_metrics = {
            "epoch": self.epoch,
            "generation": self.generation,
            "wall_clock_seconds": round(gen_elapsed, 1),
            "n_pb_runs": n_runs,
            "n_foldover_runs": len(foldover_responses),
            "n_total_runs": n_runs + len(foldover_responses),
            "best_pb_val_bpb": best_bpb if best_bpb != float("inf") else None,
            "n_significant": n_significant,
            "significant_factors": sorted(sig_factors),
            "factors_tested": factor_names,
            "se_lenth": round(se, 6),
            "screening_threshold": round(screen_t, 3),
        }
        # Save per-generation metrics
        metrics_dir = self.results_dir / "gen_metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = metrics_dir / f"e{self.epoch}_g{self.generation}.json"
        metrics_path.write_text(json.dumps(gen_metrics, indent=2, default=str))

        self.log(f"  Gen metrics: {n_significant} significant, "
                 f"{n_runs}+{len(foldover_responses)} runs, "
                 f"{gen_elapsed:.0f}s wall-clock")

        self.generation += 1
        self.save_checkpoint()  # Checkpoint after every generation for crash recovery
        return winner

    def refine_factors(self, factors: list[Factor], winner: dict) -> list[Factor]:
        """Refine factor ranges around winner for next generation.
        Skips categorical factors — they have discrete levels, not continuous ranges."""
        refined = []
        for f in factors:
            if f.name in winner and f.dtype != "categorical":
                refined.append(f.refine_around(winner[f.name]))
            else:
                refined.append(f)
        return refined

    def run_epoch(self):
        """Run K generations with adaptive factor evolution, then epoch reset."""
        self.generation = 0
        locked = self.knowledge.get_locked_factors()

        # Clear Muon-dependent locks if USE_MUON state changed
        muon_state = self.current_baseline.get("USE_MUON")
        if muon_state is not None:
            cleaned = clear_muon_dependent_locks(locked, int(muon_state))
            if len(cleaned) < len(locked):
                dropped = set(locked) - set(cleaned)
                self.log(f"  Cleared Muon-dependent locks: {dropped}")
                self.knowledge.unlock_factors(dropped)
                locked = cleaned

        # Create generation strategy (handles factor evolution between gens)
        strategy = GenerationStrategy(
            epoch=self.epoch,
            locked_factors=set(locked.keys()),
            knowledge=self.knowledge,
            max_factors=11,  # 12-run PB design
            llm_factors=self.llm_proposed_factors,
            log_fn=self.log,
        )
        self.llm_proposed_factors = None  # Consumed by strategy

        self.log(f"\n{'='*60}")
        self.log(f"EPOCH {self.epoch} START (K={self.arm_k}, adaptive strategy)")
        self.log(f"Locked factors: {list(locked.keys())}")
        self.log(f"{'='*60}")

        for g in range(self.arm_k):
            # Skip already-completed generations (resume after crash)
            if g < self.generation:
                strategy.replay_completed_gen(g, self.knowledge)
                continue

            # Get factors for this generation (evolved from prior gen's results)
            factors = strategy.select_factors_for_gen(g)
            self.log(f"  Gen {g} factors ({len(factors)}): "
                     f"{[f.name for f in factors]}")

            winner = self.run_generation(factors)

            # Record result into strategy for next generation's decisions
            latest_summary = self._get_latest_generation_summary()
            strategy.record_generation_result(g, latest_summary, winner)

            # Apply any mid-epoch locks to baseline
            mid_locks = strategy.get_mid_epoch_locks()
            if mid_locks:
                self.current_baseline.update(mid_locks)

        # FACTOR RESET -- epoch boundary
        # Update baseline with locked factor values (high confidence) only.
        # Do NOT blindly adopt epoch_best["config"] — that's a PB run with
        # half its factors at extreme values. The evolved baseline (built from
        # significant-factor-only updates across generations) is more reliable.
        epoch_best = self.knowledge.get_epoch_best(self.epoch)
        locked_factors = self.knowledge.get_locked_factors()
        for name, value in locked_factors.items():
            self.current_baseline[name] = value

        # Epoch-boundary locking (in addition to mid-epoch locks)
        pi = self.knowledge.get_active_fraction()
        lock_t = adaptive_locking_threshold(pi)
        self.log(f"  Locking threshold: π={pi:.2f} → t_lock={lock_t:.2f}")
        lock_candidates = self.knowledge.suggest_lock_candidates(locking_threshold=lock_t)
        if lock_candidates:
            to_lock = {}
            for name, effect in lock_candidates[:5]:  # Max 5 locks per epoch
                best_val = self.current_baseline.get(name)
                if best_val is not None:
                    to_lock[name] = best_val
                    self.log(f"  LOCKING {name} = {best_val} (avg effect: {effect:.6f})")
            self.knowledge.lock_factors(to_lock)

        # Graduate stale factors: high-confidence but direction-inconsistent
        # factors that have been tested extensively without ever being locked.
        # Lock them at baseline to free design slots for new factors.
        graduated = self.knowledge.graduate_stale_factors(
            self.current_baseline, min_tests=10, min_sig=5,
            log_fn=self.log,
        )
        if graduated:
            self.current_baseline.update(graduated)

        # Propose next factor set via LLM (if enabled)
        if self.llm_mode == "llm":
            try:
                from .llm_proposer import propose_factors_via_llm
                new_factors = propose_factors_via_llm(
                    self.knowledge.path, endpoint=None
                )
                if new_factors:
                    self.llm_proposed_factors = new_factors
                    self.log(f"  LLM proposed {len(new_factors)} factors for next epoch")
                else:
                    self.llm_proposed_factors = None
            except Exception as e:
                self.llm_proposed_factors = None
                self.log(f"  LLM proposal failed ({e}), using rotation")

        self.epoch += 1
        self.save_checkpoint()

        self.log(f"\nEPOCH {self.epoch - 1} COMPLETE")
        self.log(f"  Best val_bpb this epoch: {epoch_best['val_bpb']:.6f}")
        self.log(f"  Global best: {self.knowledge.get_best_val_bpb():.6f}")
        self.log(f"  Total experiments: {self.total_experiments}")
        self.log(f"  Mid-epoch locks applied: {list(strategy.get_mid_epoch_locks().keys())}")

        # --- Epoch-level retrospective metrics ---
        completed_epoch = self.epoch - 1
        epoch_metrics = {
            "epoch": completed_epoch,
            "total_locked_factors": list(self.knowledge.get_locked_factors().keys()),
            "mid_epoch_locks": list(strategy.get_mid_epoch_locks().keys()),
            "epoch_best_pb_val_bpb": epoch_best.get("val_bpb"),
            "current_baseline": dict(self.current_baseline),
        }
        # Collect per-gen metrics for retrospective K-optimization analysis
        gen_trajectory = []
        metrics_dir = self.results_dir / "gen_metrics"
        for g in range(self.arm_k):
            gm_path = metrics_dir / f"e{completed_epoch}_g{g}.json"
            if gm_path.exists():
                gm = json.loads(gm_path.read_text())
                gen_trajectory.append({
                    "gen": g,
                    "best_pb_val_bpb": gm.get("best_pb_val_bpb"),
                    "n_significant": gm.get("n_significant", 0),
                    "significant_factors": gm.get("significant_factors", []),
                    "wall_clock_s": gm.get("wall_clock_seconds", 0),
                })
        epoch_metrics["gen_trajectory"] = gen_trajectory
        epoch_metrics_path = metrics_dir / f"epoch_{completed_epoch}_summary.json"
        epoch_metrics_path.write_text(json.dumps(epoch_metrics, indent=2, default=str))

        # Log generation trajectory for quick visual
        self.log(f"  Generation trajectory:")
        for gt in gen_trajectory:
            bpb = gt["best_pb_val_bpb"]
            bpb_str = f"{bpb:.4f}" if bpb else "N/A"
            self.log(f"    Gen {gt['gen']}: best={bpb_str} "
                     f"({gt['n_significant']} sig: {gt['significant_factors']}) "
                     f"[{gt['wall_clock_s']:.0f}s]")

        # POST-EPOCH VALIDATION: run the evolved baseline solo to measure true val_bpb
        self._run_epoch_validation(completed_epoch)

    def _run_epoch_validation(self, completed_epoch: int):
        """Run a single solo experiment with the evolved baseline config.

        Called after every epoch (G3 complete, before next epoch starts).
        Runs ONE experiment with NO parallelism so the result reflects true
        solo performance — comparable to leaderboard submissions.

        Architecture factors (HEAD_DIM, ASPECT_RATIO) are fixed at model_dim=256
        to avoid the throughput-capacity confound discovered in PB screening.
        """
        self.log(f"\n--- EPOCH {completed_epoch} VALIDATION RUN ---")

        # Build validation config from current evolved baseline
        val_config = dict(self.current_baseline)

        # FIX architecture confound: force model_dim=256
        # HEAD_DIM and ASPECT_RATIO change model size, which confounds
        # throughput with capacity in fixed-time training budgets.
        val_config["HEAD_DIM"] = 128
        val_config["ASPECT_RATIO"] = 64
        # Ensure MLP hidden dim is even at model_dim=256
        mlp_exp = float(val_config.get("MLP_EXPANSION", 4.0))
        if int(mlp_exp * 256) % 2 != 0:
            val_config["MLP_EXPANSION"] = 4.0
            self.log(f"  Validation: MLP_EXPANSION→4.0 (hidden dim even fix)")

        self.log(f"  Validation config: {json.dumps(val_config, indent=2, default=str)}")

        # Prepare validation script
        val_path = self.workspace / "train_validation.py"
        shutil.copy2(self.train_py, val_path)
        apply_config(val_config, val_path)

        # Run SOLO (single process, full GPU)
        self.log(f"  Launching solo validation (no parallelism)...")
        proc = subprocess.Popen(
            ["uv", "run", str(val_path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            cwd=str(self.workspace),
            env={**os.environ, "PYTHONPATH": str(self.workspace)},
        )

        try:
            stdout, stderr = proc.communicate(timeout=RUN_TIMEOUT)
            val_bpb = self._parse_val_bpb(stdout)
            peak_vram = self._parse_peak_vram(stdout)

            if val_bpb is None or (val_bpb is not None and val_bpb != val_bpb):
                self.log(f"  VALIDATION FAILED (NaN or crash)")
                if stderr:
                    self.log(f"  stderr: {stderr[-300:]}")
                val_bpb = float("inf")
            else:
                self.log(f"  *** VALIDATION RESULT: val_bpb = {val_bpb:.6f} ***")
                self.log(f"  Peak VRAM: {peak_vram:.0f}MB")

                # Compare against leaderboard target
                target = 1.294
                if val_bpb < target:
                    self.log(f"  🏆 BEATS LEADERBOARD TARGET ({target})!")
                elif val_bpb < 1.35:
                    self.log(f"  Close to target ({target}), gap: {val_bpb - target:.4f}")
                else:
                    self.log(f"  Gap to target ({target}): {val_bpb - target:.4f}")

                # Record validation result
                global_best = self.knowledge.get_best_val_bpb()
                if val_bpb < global_best:
                    self.log(f"  NEW GLOBAL BEST (was {global_best:.6f})")

                # Log to TSV
                with open(self.results_tsv, "a", newline="") as f:
                    writer = csv.writer(f, delimiter="\t")
                    writer.writerow([
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        self.arm_name, completed_epoch, "VAL", "validation_solo",
                        f"{val_bpb:.6f}",
                        f"{peak_vram:.0f}" if peak_vram else "0",
                        "ok", "EPOCH_VALIDATION",
                    ])

                # Store validation result in knowledge
                self.knowledge.record_validation(
                    completed_epoch, val_bpb, val_config
                )

        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            self.log(f"  VALIDATION TIMEOUT ({RUN_TIMEOUT}s)")

        finally:
            val_path.unlink(missing_ok=True)

        self.total_experiments += 1
        self.save_checkpoint()

    def _get_latest_generation_summary(self) -> dict:
        """Retrieve the most recently recorded generation summary from knowledge."""
        if self.epoch < len(self.knowledge.data.get("epochs", [])):
            epoch_data = self.knowledge.data["epochs"][self.epoch]
            gens = epoch_data.get("generations", [])
            if gens:
                return gens[-1]
        return {}

    def run_forever(self):
        """Main autonomous loop. Runs until killed."""
        self.resume_if_checkpoint()
        while True:
            self.run_epoch()

    def save_checkpoint(self):
        """Save state for crash recovery."""
        checkpoint = {
            "arm_k": self.arm_k,
            "arm_name": self.arm_name,
            "epoch": self.epoch,
            "generation": self.generation,
            "current_baseline": self.current_baseline,
            "total_experiments": self.total_experiments,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.checkpoint_path.write_text(json.dumps(checkpoint, indent=2, default=str))

    def resume_if_checkpoint(self):
        """Resume from checkpoint if one exists."""
        if self.checkpoint_path.exists():
            checkpoint = json.loads(self.checkpoint_path.read_text())
            self.epoch = checkpoint.get("epoch", 0)
            self.generation = checkpoint.get("generation", 0)
            self.current_baseline = checkpoint.get("current_baseline", self.current_baseline)
            self.total_experiments = checkpoint.get("total_experiments", 0)
            self.log(f"Resumed from checkpoint: epoch={self.epoch}, "
                     f"experiments={self.total_experiments}")


def setup_workspace(workspace_dir: Path, source_dir: Path):
    """Set up an isolated workspace with copies of train.py and prepare.py."""
    workspace_dir.mkdir(parents=True, exist_ok=True)
    for filename in ["train.py", "prepare.py"]:
        src = source_dir / filename
        dst = workspace_dir / filename
        if src.exists():
            shutil.copy2(src, dst)


def run_arm(arm_k: int, arm_name: str, baseline: dict,
            workspace: Path, results_dir: Path, llm_mode: str,
            max_parallel: int = 0):
    """Entry point for a worker process."""
    worker = ArmWorker(
        arm_k=arm_k, arm_name=arm_name, baseline_config=baseline,
        workspace_dir=workspace, results_dir=results_dir, llm_mode=llm_mode,
        max_parallel=max_parallel,
    )
    worker.run_forever()


def read_best_val_bpb(tsv_path: Path) -> float:
    """Read the best val_bpb from a results TSV."""
    best = float("inf")
    if not tsv_path.exists():
        return best
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                val = float(row.get("val_bpb", "inf"))
                if val < best:
                    best = val
            except (ValueError, TypeError):
                pass
    return best


def log_comparison(results_dir: Path):
    """Log comparative status of both arms."""
    best_a = read_best_val_bpb(results_dir / "arm_a.tsv")
    best_b = read_best_val_bpb(results_dir / "arm_b.tsv")
    elapsed = time.strftime("%H:%M:%S")
    leader = "A (K=3)" if best_a <= best_b else "B (K=4)"
    print(f"\n[{elapsed}] Arm A (K=3): {best_a:.6f} | "
          f"Arm B (K=4): {best_b:.6f} | Leader: {leader}\n", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Factorial-evolutionary autonomous ML research"
    )
    parser.add_argument("--llm", action="store_true",
                        help="Use LLM for factor proposals at epoch boundaries")
    parser.add_argument("--single", type=int, default=None, metavar="K",
                        help="Run a single arm with K generations per epoch")
    parser.add_argument("--convergence-hours", type=float, default=CONVERGENCE_HOURS,
                        help=f"Hours before converging on winning K (default: {CONVERGENCE_HOURS})")
    parser.add_argument("--workspace", type=str, default=None,
                        help="Override workspace directory")
    parser.add_argument("--max-parallel", type=int, default=0, metavar="N",
                        help="Max parallel experiments per arm (0=auto-detect from memory)")
    args = parser.parse_args()

    llm_mode = "llm" if args.llm else "rotation"
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Read baseline config from train.py
    baseline = read_current_config(PROJECT_ROOT / "train.py")
    print(f"Baseline config: {json.dumps(baseline, indent=2, default=str)}")

    if args.single is not None:
        # Single-arm mode
        arm_k = args.single
        arm_name = f"arm_k{arm_k}"
        ws = Path(args.workspace) if args.workspace else PROJECT_ROOT / "workspaces" / arm_name
        setup_workspace(ws, PROJECT_ROOT)
        print(f"Running single arm K={arm_k} in {ws}")
        run_arm(arm_k, arm_name, baseline, ws, results_dir, llm_mode,
                args.max_parallel)
        return

    # Dual-arm mode with convergence
    ws_a = PROJECT_ROOT / "workspaces" / "arm_a"
    ws_b = PROJECT_ROOT / "workspaces" / "arm_b"
    setup_workspace(ws_a, PROJECT_ROOT)
    setup_workspace(ws_b, PROJECT_ROOT)

    parallel_budget = args.max_parallel or ArmWorker._detect_parallel_budget()
    print(f"Launching parallel arms: A (K=3) and B (K=4)")
    print(f"Convergence after {args.convergence_hours} hours")
    print(f"LLM mode: {llm_mode}")
    print(f"Parallel budget: {parallel_budget} experiments per arm")
    print(f"Workspaces: {ws_a}, {ws_b}")
    print()

    arm_a = mp.Process(
        target=run_arm,
        args=(3, "arm_a", baseline, ws_a, results_dir, llm_mode,
              args.max_parallel),
    )
    arm_b = mp.Process(
        target=run_arm,
        args=(4, "arm_b", baseline, ws_b, results_dir, llm_mode,
              args.max_parallel),
    )

    arm_a.start()
    arm_b.start()

    start_time = time.time()
    converged = False

    try:
        while arm_a.is_alive() or arm_b.is_alive():
            time.sleep(300)  # Check every 5 min
            elapsed_hours = (time.time() - start_time) / 3600

            log_comparison(results_dir)

            if not converged and elapsed_hours >= args.convergence_hours:
                best_a = read_best_val_bpb(results_dir / "arm_a.tsv")
                best_b = read_best_val_bpb(results_dir / "arm_b.tsv")

                # Need at least some results from both arms
                if best_a == float("inf") or best_b == float("inf"):
                    print(f"[{elapsed_hours:.1f}h] Waiting for both arms to produce results...")
                    continue

                winner_k = 3 if best_a < best_b else 4
                loser_proc = arm_b if winner_k == 3 else arm_a
                loser_name = "arm_b" if winner_k == 3 else "arm_a"
                winner_name = "arm_a" if winner_k == 3 else "arm_b"

                print(f"\n{'='*60}")
                print(f"CONVERGENCE at {elapsed_hours:.1f}h")
                print(f"Arm A (K=3): {best_a:.6f}")
                print(f"Arm B (K=4): {best_b:.6f}")
                print(f"Winner: K={winner_k}")
                print(f"Killing {loser_name}, spawning 2nd worker on K={winner_k}")
                print(f"{'='*60}\n")

                loser_proc.terminate()
                loser_proc.join(timeout=30)

                # Load winner's current best config
                winner_knowledge = KnowledgeStore(results_dir / f"{winner_name}_knowledge.json")
                winner_config = winner_knowledge.get_best_config()
                merged_baseline = {**baseline, **winner_config}

                # Set up new workspace for the repurposed worker
                loser_ws = ws_b if winner_k == 3 else ws_a
                setup_workspace(loser_ws, PROJECT_ROOT)
                # Apply winner's config to the new workspace
                apply_config(merged_baseline, loser_ws / "train.py")

                new_worker = mp.Process(
                    target=run_arm,
                    args=(winner_k, loser_name, merged_baseline,
                          loser_ws, results_dir, llm_mode,
                          args.max_parallel),
                )
                new_worker.start()

                if winner_k == 3:
                    arm_b = new_worker
                else:
                    arm_a = new_worker

                converged = True
                print(f"Both workers now running K={winner_k}.\n")

    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        arm_a.terminate()
        arm_b.terminate()
        arm_a.join(timeout=10)
        arm_b.join(timeout=10)

    # Final report
    print(f"\n{'='*60}")
    print("FINAL REPORT")
    print(f"{'='*60}")
    best_a = read_best_val_bpb(results_dir / "arm_a.tsv")
    best_b = read_best_val_bpb(results_dir / "arm_b.tsv")
    print(f"Arm A: best val_bpb = {best_a:.6f}")
    print(f"Arm B: best val_bpb = {best_b:.6f}")
    print(f"Overall best: {min(best_a, best_b):.6f}")

    # Load and display best config
    for arm in ["arm_a", "arm_b"]:
        ks = KnowledgeStore(results_dir / f"{arm}_knowledge.json")
        if ks.get_best_val_bpb() < float("inf"):
            print(f"\n{arm} best config:")
            print(json.dumps(ks.get_best_config(), indent=2, default=str))
            print(f"Total experiments: {ks.total_experiments}")


if __name__ == "__main__":
    main()
