"""Experiment 2: Knowledge Consolidation — Real embedding + LLM comparison.

Uses BGE-small-zh from local LM Studio for candidate filtering.
Compares GLM-5.1 on merge judgment quality.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from hermes_core.api import get_all_providers, PROVIDERS
from hermes_core.consolidation import (
    ConsolidationEngine,
    generate_consolidation_corpus,
    build_ground_truth,
)


LM_STUDIO_URL = "http://127.0.0.1:1234/v1/embeddings"
EMBEDDING_MODEL = "text-embedding-bge-small-zh-v1.5"


def get_embedding(text: str) -> np.ndarray:
    resp = requests.post(LM_STUDIO_URL, json={"model": EMBEDDING_MODEL, "input": text})
    data = resp.json()
    if "data" in data:
        return np.array(data["data"][0]["embedding"], dtype=np.float32)
    raise ValueError(f"Embedding error: {data}")


def run():
    print("=" * 60)
    print("EXPERIMENT 2: Knowledge Consolidation (Real Embeddings)")
    print("=" * 60)

    results_dir = Path(__file__).parent.parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Generate data
    corpus = generate_consolidation_corpus(50)
    ground_truth = build_ground_truth(corpus)
    gt_set = {(p["id_a"], p["id_b"]) for p in ground_truth}

    # Compute real embeddings
    print(f"Computing embeddings for {len(corpus)} records...")
    embeddings = {}
    for i, r in enumerate(corpus):
        embeddings[r["id"]] = get_embedding(r["content"])
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(corpus)} done")

    # Quick similarity sanity check
    from hermes_core.consolidation import ConsolidationEngine
    redis_ids = [r["id"] for r in corpus if r["cluster"] == "Redis"]
    api_ids = [r["id"] for r in corpus if r["cluster"] == "API 设计"]
    cross_sim = float(np.dot(embeddings[redis_ids[0]], embeddings[api_ids[0]]) /
                      (np.linalg.norm(embeddings[redis_ids[0]]) * np.linalg.norm(embeddings[api_ids[0]])))
    intra_sims = []
    for i in range(min(5, len(redis_ids))):
        for j in range(i+1, min(5, len(redis_ids))):
            sim = float(np.dot(embeddings[redis_ids[i]], embeddings[redis_ids[j]]) /
                        (np.linalg.norm(embeddings[redis_ids[i]]) * np.linalg.norm(embeddings[redis_ids[j]])))
            intra_sims.append(sim)
    avg_intra = sum(intra_sims) / len(intra_sims)
    print(f"\nEmbedding quality check:")
    print(f"  Cross-cluster sim (Redis vs API): {cross_sim:.4f}")
    print(f"  Intra-cluster avg sim (Redis):     {avg_intra:.4f}")
    print(f"  Discrimination: {'GOOD' if avg_intra > cross_sim + 0.1 else 'WEAK'}")

    # Define embedding function for ConsolidationEngine
    def emb_fn(text: str) -> np.ndarray:
        # Look up pre-computed embedding by content match
        for r in corpus:
            if r["content"] == text:
                return embeddings[r["id"]]
        # Fallback: compute on the fly
        return get_embedding(text)

    results = {}

    # --- Control A: pure embedding at various thresholds ---
    for threshold in [0.70, 0.80, 0.85, 0.90, 0.95]:
        engine = ConsolidationEngine(emb_fn, llm_fn=None, similarity_threshold=0.50)
        for r in corpus:
            engine.add_record(r["id"], r["content"], {"cluster": r["cluster"]})

        merge_set = set()
        ids = list(engine.records.keys())
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                sim = engine.cosine_sim(engine.records[ids[i]]["embedding"], engine.records[ids[j]]["embedding"])
                if sim >= threshold:
                    pair = (min(ids[i], ids[j]), max(ids[i], ids[j]))
                    merge_set.add(pair)

        tp = len(merge_set & gt_set)
        fp = len(merge_set - gt_set)
        fn = len(gt_set - merge_set)
        p = tp / max(tp + fp, 1)
        rc = tp / max(tp + fn, 1)
        print(f"\n  Embedding @ {threshold}: candidates={len(merge_set)}, TP={tp}, FP={fp}, P={p:.2%}, R={rc:.2%}")
        results[f"embedding_{threshold}"] = {
            "strategy": f"embedding_{threshold}",
            "candidates": len(merge_set),
            "true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "precision": round(p, 4), "recall": round(rc, 4),
        }

    # --- LLM-driven merge (per provider) ---
    providers = get_all_providers()
    sim_threshold = results.get("embedding_0.85", {}).get("precision", 0) > 0.8 and 0.60 or 0.50

    for provider_key, llm_bench in providers.items():
        print(f"\n--- LLM merge: {llm_bench.config.name} ---")
        engine = ConsolidationEngine(emb_fn, llm_fn=llm_bench, similarity_threshold=0.50)
        for r in corpus:
            engine.add_record(r["id"], r["content"], {"cluster": r["cluster"]})

        llm_results = engine.judge_all_candidates()
        merge_set = {(r["id_a"], r["id_b"]) for r in llm_results if r.get("should_merge")}
        tp = len(merge_set & gt_set)
        fp = len(merge_set - gt_set)
        fn = len(gt_set - merge_set)
        p = tp / max(tp + fp, 1)
        rc = tp / max(tp + fn, 1)

        print(f"  Candidates: {len(llm_results)}, Merged: {len(merge_set)}")
        print(f"  TP: {tp}, FP: {fp}, FN: {fn}")
        print(f"  Precision: {p:.2%}, Recall: {rc:.2%}")
        print(f"  LLM stats: {llm_bench.stats.to_dict()}")

        results[f"llm_{provider_key}"] = {
            "strategy": f"llm_{provider_key}",
            "model": llm_bench.config.name,
            "candidates": len(llm_results),
            "merged": len(merge_set),
            "true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "precision": round(p, 4), "recall": round(rc, 4),
            "llm_stats": llm_bench.stats.to_dict(),
        }

    # Save
    out = {
        "experiment": "exp2_consolidation_v2",
        "embedding_model": EMBEDDING_MODEL,
        "params": {"n_records": len(corpus), "similarity_threshold": 0.50},
        "embedding_quality": {"cross_cluster_sim": round(cross_sim, 4), "intra_cluster_avg_sim": round(avg_intra, 4)},
        "results": results,
    }
    with open(results_dir / "exp2_results_v2.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Comparison table
    print(f"\n{'Strategy':<25} {'Cand':>5} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Calls':>6}")
    print("-" * 60)
    for key, r in results.items():
        p, rc = r["precision"], r["recall"]
        f1 = 2 * p * rc / max(p + rc, 1e-8)
        calls = r.get("llm_stats", {}).get("calls", 0)
        print(f"{key:<25} {r.get('candidates',0):>5} {p:>8.2%} {rc:>8.2%} {f1:>8.2%} {calls:>6}")

    print(f"\nResults saved to {results_dir / 'exp2_results_v2.json'}")
    return out


if __name__ == "__main__":
    run()
