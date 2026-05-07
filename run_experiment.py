"""Experiment runner — execute and record experiment results.

Usage:
    python run_experiment.py --exp 1        # Run single experiment
    python run_experiment.py --exp all      # Run all experiments
    python run_experiment.py --exp 1 --report  # Run and generate report
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_exp1():
    """Experiment 1: Lifecycle state machine."""
    from hermes_core.lifecycle import LifecycleManager, generate_test_data, simulate_timeline, build_reference_schedule

    print("=" * 60)
    print("EXPERIMENT 1: Lifecycle State Machine")
    print("=" * 60)

    # Generate test data
    records = generate_test_data(n=1000, days_span=180)
    ref_schedule = build_reference_schedule(records)
    print(f"Generated {len(records)} test records, {sum(len(v) for v in ref_schedule.values())} reference events")

    # Run Hermes state machine
    db_hermes = str(RESULTS_DIR / "exp1_hermes.db")
    mgr_hermes = LifecycleManager(db_hermes, stale_days=30, archive_days=90)
    for r in records:
        mgr_hermes.add_record(r["record_id"], r["content"], r["created_at"], r.get("pinned", False))
    hermes_timeseries = simulate_timeline(mgr_hermes, days=180, reference_schedule=ref_schedule)

    # Run control A: simple expiry
    from hermes_core.lifecycle import SimpleExpiryManager
    db_simple = str(RESULTS_DIR / "exp1_simple.db")
    mgr_simple = SimpleExpiryManager(db_simple, expire_days=30)
    for r in records:
        mgr_simple.add_record(r["record_id"], r["content"], r["created_at"])
    simple_timeseries = simulate_timeline(mgr_simple, days=180, reference_schedule=ref_schedule)

    # Run control B: no management
    from hermes_core.lifecycle import NoOpManager
    db_noop = str(RESULTS_DIR / "exp1_noop.db")
    mgr_noop = NoOpManager(db_noop)
    for r in records:
        mgr_noop.add_record(r["record_id"], r["content"], r["created_at"])
    noop_timeseries = simulate_timeline(mgr_noop, days=180)

    # Save results
    results = {
        "experiment": "exp1_lifecycle",
        "timestamp": datetime.now().isoformat(),
        "params": {"n_records": 1000, "days": 180, "stale_days": 30, "archive_days": 90},
        "hermes": hermes_timeseries,
        "simple_expiry": simple_timeseries,
        "no_management": noop_timeseries,
    }

    out_path = RESULTS_DIR / "exp1_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    h_final = hermes_timeseries[-1]
    s_final = simple_timeseries[-1]
    n_final = noop_timeseries[-1]

    print(f"\n{'Metric':<25} {'Hermes':>10} {'Simple':>10} {'No-Op':>10}")
    print("-" * 55)
    print(f"{'active_ratio':.<25} {h_final['active_ratio']:>10.1%} {s_final['active_ratio']:>10.1%} {n_final['active_ratio']:>10.1%}")
    print(f"{'total_transitions':.<25} {h_final['total_transitions']:>10} {s_final['total_transitions']:>10} {s_final['total_transitions']:>10}")
    if "reactivations" in h_final:
        print(f"{'reactivations':.<25} {h_final['reactivations']:>10} {'N/A':>10} {'N/A':>10}")

    print(f"\nResults saved to {out_path}")
    return results


def run_exp2():
    """Experiment 2: Knowledge consolidation."""
    print("=" * 60)
    print("EXPERIMENT 2: Knowledge Consolidation")
    print("=" * 60)
    print("Requires LLM API. Run with: python experiments/exp2-consolidation/run.py")
    print("Skipping (needs API configuration).")
    return None


def run_exp3():
    """Experiment 3: Context prefetch."""
    print("=" * 60)
    print("EXPERIMENT 3: Context Prefetch")
    print("=" * 60)
    print("Requires LLM API. Run with: python experiments/exp3-prefetch/run.py")
    print("Skipping (needs API configuration).")
    return None


def run_exp4():
    """Experiment 4: Context compression."""
    print("=" * 60)
    print("EXPERIMENT 4: Context Compression")
    print("=" * 60)
    print("Requires LLM API. Run with: python experiments/exp4-compression/run.py")
    print("Skipping (needs API configuration).")
    return None


EXPERIMENTS = {1: run_exp1, 2: run_exp2, 3: run_exp3, 4: run_exp4}


def main():
    parser = argparse.ArgumentParser(description="Hermes mechanism experiments")
    parser.add_argument("--exp", type=str, required=True, help="Experiment number (1-4) or 'all'")
    parser.add_argument("--report", action="store_true", help="Generate report after running")
    args = parser.parse_args()

    os.chdir(Path(__file__).parent)
    sys.path.insert(0, str(Path(__file__).parent))

    if args.exp == "all":
        for n in sorted(EXPERIMENTS):
            EXPERIMENTS[n]()
    else:
        n = int(args.exp)
        if n not in EXPERIMENTS:
            print(f"Unknown experiment: {n}")
            sys.exit(1)
        result = EXPERIMENTS[n]()
        if args.report and result:
            from hermes_core.report import generate_report
            generate_report(n, result)


if __name__ == "__main__":
    main()
