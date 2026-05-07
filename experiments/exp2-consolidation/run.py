"""Experiment 2: Knowledge Consolidation — LLM-driven merge quality.

Compares:
  - Strategy: Hermes LLM merge vs pure embedding threshold
  - LLM: GLM-5.1
  - Ground truth: same-cluster = should merge
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from hermes_core.api import get_all_providers, PROVIDERS
from hermes_core.consolidation import (
    ConsolidationEngine,
    generate_consolidation_corpus,
    build_ground_truth,
)
from hermes_core.prefetch import MockEmbedding
import numpy as np


def run():
    print("=" * 60)
    print("EXPERIMENT 2: Knowledge Consolidation")
    print("=" * 60)

    results_dir = Path(__file__).parent.parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Generate data
    corpus = generate_consolidation_corpus(50)  # 10 per cluster x 5
    ground_truth = build_ground_truth(corpus)
    gt_set = {(p["id_a"], p["id_b"]) for p in ground_truth}

    print(f"Corpus: {len(corpus)} records, {len(ground_truth)} ground-truth merge pairs")
    print(f"Providers: {list(PROVIDERS.keys())}")

    results = {}

    # --- Control: pure embedding merge (no LLM) ---
    print("\n--- Control: Pure embedding merge (threshold=0.85) ---")
    emb = MockEmbedding(128)
    engine_ctrl = ConsolidationEngine(emb, llm_fn=None, similarity_threshold=0.60)
    for r in corpus:
        engine_ctrl.add_record(r["id"], r["content"], {"cluster": r["cluster"]})
    ctrl_results = engine_ctrl.pure_embedding_merge(merge_threshold=0.85)
    ctrl_set = {(r["id_a"], r["id_b"]) for r in ctrl_results}
    ctrl_tp = len(ctrl_set & gt_set)
    ctrl_fp = len(ctrl_set - gt_set)
    ctrl_fn = len(gt_set - ctrl_set)
    ctrl_precision = ctrl_tp / max(ctrl_tp + ctrl_fp, 1)
    ctrl_recall = ctrl_tp / max(ctrl_tp + ctrl_fn, 1)

    print(f"  Candidates: {len(ctrl_results)}, TP: {ctrl_tp}, FP: {ctrl_fp}, FN: {ctrl_fn}")
    print(f"  Precision: {ctrl_precision:.2%}, Recall: {ctrl_recall:.2%}")

    results["embedding_only"] = {
        "strategy": "embedding_threshold_0.85",
        "total_candidates": len(ctrl_results),
        "true_positives": ctrl_tp,
        "false_positives": ctrl_fp,
        "false_negatives": ctrl_fn,
        "precision": round(ctrl_precision, 4),
        "recall": round(ctrl_recall, 4),
        "llm_stats": {"calls": 0},
    }

    # --- Experiment: LLM-driven merge (per provider) ---
    providers = get_all_providers()

    for provider_key, llm_bench in providers.items():
        print(f"\n--- LLM merge: {llm_bench.config.name} ---")
        emb2 = MockEmbedding(128)
        engine_llm = ConsolidationEngine(emb2, llm_fn=llm_bench, similarity_threshold=0.60)
        for r in corpus:
            engine_llm.add_record(r["id"], r["content"], {"cluster": r["cluster"]})

        llm_results = engine_llm.judge_all_candidates()
        llm_merge_set = {(r["id_a"], r["id_b"]) for r in llm_results if r.get("should_merge")}
        llm_tp = len(llm_merge_set & gt_set)
        llm_fp = len(llm_merge_set - gt_set)
        llm_fn = len(gt_set - llm_merge_set)
        llm_precision = llm_tp / max(llm_tp + llm_fp, 1)
        llm_recall = llm_tp / max(llm_tp + llm_fn, 1)

        print(f"  Candidates: {len(llm_results)}, Merged: {len(llm_merge_set)}")
        print(f"  TP: {llm_tp}, FP: {llm_fp}, FN: {llm_fn}")
        print(f"  Precision: {llm_precision:.2%}, Recall: {llm_recall:.2%}")
        print(f"  LLM stats: {llm_bench.stats.to_dict()}")

        results[f"llm_{provider_key}"] = {
            "strategy": f"llm_merge_{provider_key}",
            "model": llm_bench.config.name,
            "total_candidates": len(llm_results),
            "merged": len(llm_merge_set),
            "true_positives": llm_tp,
            "false_positives": llm_fp,
            "false_negatives": llm_fn,
            "precision": round(llm_precision, 4),
            "recall": round(llm_recall, 4),
            "llm_stats": llm_bench.stats.to_dict(),
        }

    # Save
    out = {
        "experiment": "exp2_consolidation",
        "params": {"n_records": len(corpus), "similarity_threshold": 0.60, "merge_threshold": 0.85},
        "results": results,
    }
    with open(results_dir / "exp2_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Print comparison table
    print(f"\n{'Strategy':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'LLM Calls':>10}")
    print("-" * 65)
    for key, r in results.items():
        p, rc = r["precision"], r["recall"]
        f1 = 2 * p * rc / max(p + rc, 1e-8)
        calls = r["llm_stats"].get("calls", 0)
        print(f"{key:<25} {p:>10.2%} {rc:>10.2%} {f1:>10.2%} {calls:>10}")

    print(f"\nResults saved to {results_dir / 'exp2_results.json'}")
    return out


if __name__ == "__main__":
    run()
