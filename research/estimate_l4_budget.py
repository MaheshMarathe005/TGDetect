"""L4 GPU budget + statistical-power estimator for the DEDALE (or any) APT study.

Every timing constant below is anchored on THIS project's own measured L4 runs, so
the estimate is grounded, not guessed. Sources:

  * Windows-APT (batch 256, 200 epochs, 61,206 train events => 48,000 steps/run,
    10 workers sharing one L4): GNN+rehearsal 783.50 s/seed; full hybrid
    TGNN+GRU+attention+CNN+rehearsal 1,947.54 s/seed. Ten runs finished in ~33 min
    wall-clock. (research/findings/windows_apt_dataset_analysis.md, STATUS.md)
  * Component study (9,830 train events => 7,800 steps/run, 200 epochs): GNN 148.14 s,
    plain TGNN 156.72 s, TGNN+GRU+attention 275.49 s. (TGNN_COMPONENT_FINDINGS.md)
  * Aggregate throughput calibration on ONE L4 (hybrid workload):
    1/2/4/8/10 workers -> 35.76/47.94/99.77/201.71/249.20 steps/s. (STATUS.md)

Nothing here launches training. It only prints how much L4 time and money a
statistically conclusive comparison would need. Run:  python estimate_l4_budget.py
"""
from __future__ import annotations

import argparse
import math

# --- Measured per-optimizer-step wall-clock, batch 256, under ~10-worker sharing ---
# Windows-APT is the primary anchor: same batch size, 200 epochs, and it is the exact
# hybrid the user built. Derived as seconds_per_seed / steps_per_run.
MS_PER_STEP = {
    # model_key: (milliseconds per step, approx parameter count)
    "feature_only": (0.0, 0),           # logistic / boosted trees: CPU, ~free vs L4
    "gnn_rehearsal": (783.50 / 48000 * 1000, 43394),         # 16.32 ms
    "tgnn_time": (156.72 / 7800 * 1000, 35714),              # 20.09 ms (plain TGNN)
    "tgnn_gru_attn": (275.49 / 7800 * 1000, 183938),         # 35.32 ms
    "hybrid_gru_attn_cnn": (1947.54 / 48000 * 1000, 290434), # 40.57 ms (user's model)
}

# Effective worker concurrency achieved on one L4 (measured: 13,655 s of summed
# per-run work finished in ~33 min => ~6.9x). Use 7 as the grounded value.
EFFECTIVE_WORKERS = 7.0
BATCH = 256

# z-values for a two-sided paired test.
Z_ALPHA = 1.960   # alpha = 0.05
Z_POWER = 0.842   # power = 0.80

def steps_per_run(train_events: int, epochs: int) -> int:
    return math.ceil(train_events / BATCH) * epochs


def seed_run_seconds(model_key: str, train_events: int, epochs: int) -> float:
    ms, _ = MS_PER_STEP[model_key]
    return steps_per_run(train_events, epochs) * ms / 1000.0


def seeds_for_power(delta_points: float, paired_sd_points: float) -> int:
    """Paired-difference seeds to detect a true F1 gap `delta_points` at a=.05, power=.80."""
    if delta_points <= 0:
        raise ValueError("delta must be positive")
    n = ((Z_ALPHA + Z_POWER) * paired_sd_points / delta_points) ** 2
    return max(2, math.ceil(n))


def expected_paired_sd(measured_sd_25pos: float, test_positives: int) -> float:
    """Test-set sampling noise shrinks like 1/sqrt(positives), plus an irreducible
    optimization-seed floor. Anchored on the observed ~5-7 pt paired SD at 25 positives."""
    sampling = measured_sd_25pos * math.sqrt(25.0 / max(1, test_positives))
    optimization_floor = 0.7  # residual seed-to-seed SD even with a huge test set
    return math.hypot(sampling, optimization_floor)


def budget(models, train_events, epochs, seeds, l4_rate):
    total_seconds = sum(seed_run_seconds(m, train_events, epochs) * seeds for m in models)
    wall_hours = total_seconds / EFFECTIVE_WORKERS / 3600.0
    return {
        "runs": len(models) * seeds,
        "steps_per_run": steps_per_run(train_events, epochs),
        "compute_gpu_hours": total_seconds / 3600.0,   # summed single-stream work
        "wall_clock_hours_1xL4": wall_hours,
        "billable_l4_hours": wall_hours,  # one physical L4 => billable == wall-clock
        "cost": wall_hours * l4_rate,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=500_000, help="total DEDALE events (guess if unknown)")
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--test-positives", type=int, default=2500)
    ap.add_argument("--delta", type=float, default=1.0, help="F1 gap to detect, points")
    ap.add_argument("--sd25", type=float, default=6.0, help="paired SD at 25 positives (observed 5-7)")
    ap.add_argument("--l4-rate", type=float, default=0.80, help="USD/L4-hour; VERIFY at modal.com/pricing")
    args = ap.parse_args()

    train_events = int(args.events * args.train_frac)
    sd = expected_paired_sd(args.sd25, args.test_positives)
    seeds = seeds_for_power(args.delta, sd)

    print(f"DEDALE plan: {args.events:,} events, train={train_events:,}, epochs={args.epochs}")
    print(f"Expected paired-F1 SD at {args.test_positives:,} positives: {sd:.2f} pts "
          f"(was ~{args.sd25:.0f} pts at 25 positives)")
    print(f"Seeds to detect a {args.delta:.1f}-pt gap (a=.05, power=.80): {seeds}\n")

    print(f"{'model':22} {'params':>8} {'ms/step':>8} {'sec/seed':>9}")
    for m in MS_PER_STEP:
        ms, p = MS_PER_STEP[m]
        print(f"{m:22} {p:>8,} {ms:>8.2f} {seed_run_seconds(m, train_events, args.epochs):>9.0f}")

    print()
    two = ["gnn_rehearsal", "hybrid_gru_attn_cnn"]
    grid = ["gnn_rehearsal", "tgnn_time", "tgnn_gru_attn", "hybrid_gru_attn_cnn"]
    for label, models in (("2-model replication (GNN vs hybrid)", two),
                          ("4-arm ablation (isolate each component)", grid)):
        b = budget(models, train_events, args.epochs, seeds, args.l4_rate)
        print(f"[{label}]  {b['runs']} runs")
        print(f"   billable L4-hours : {b['billable_l4_hours']:.1f}")
        print(f"   wall-clock on 1 L4: {b['wall_clock_hours_1xL4']:.1f} h")
        print(f"   cost @ ${args.l4_rate:.2f}/h : ${b['cost']:.0f}  (verify current Modal L4 rate)\n")


if __name__ == "__main__":
    main()
