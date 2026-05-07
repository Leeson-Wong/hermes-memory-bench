"""Experiment 3 v2: Prefetch — revised with random-K baseline and cost tracking.

Changes from v1:
  - Added random-5 baseline (proves embedding-based prefetch value)
  - Only GLM (removed second model)
  - Track injected token count per strategy
  - Track embedding computation cost

Strategies:
  1. none:    No context (control)
  2. full:    Inject all 50 records (upper bound, unrealistic cost)
  3. prefetch: Embedding-based top-5 (Hermes mechanism)
  4. random:  Random 5 records (proves embedding adds value over naive selection)

Usage:
    cd F:/mime/learn
    python -m experiments.exp3-prefetch.run_v2
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import requests

from hermes_core.api import LLMBenchmark, PROVIDERS

LM_STUDIO_URL = "http://127.0.0.1:1234/v1/embeddings"
EMBEDDING_MODEL = "text-embedding-bge-small-zh-v1.5"

SCENARIOS = [
    {"question": "我们要给电商系统加购物车功能，之前有什么相关决策？", "relevant_topics": ["Redis", "API 设计", "数据库"]},
    {"question": "用户反馈登录经常超时，之前认证系统做过什么优化？", "relevant_topics": ["认证系统", "Redis", "API 设计"]},
    {"question": "我们需要支持百万级并发，之前的缓存策略还适用吗？", "relevant_topics": ["Redis", "数据库", "CI/CD"]},
    {"question": "新项目要选 API 框架，之前 REST 和 GraphQL 怎么选的？", "relevant_topics": ["API 设计", "认证系统"]},
    {"question": "数据库慢查询越来越多，之前做过什么优化？", "relevant_topics": ["数据库", "Redis"]},
    {"question": "要加灰度发布能力，之前的 CI/CD 流程怎么改造？", "relevant_topics": ["CI/CD", "API 设计"]},
    {"question": "需要给开放平台设计鉴权，之前的 OAuth 方案能复用吗？", "relevant_topics": ["认证系统", "API 设计"]},
    {"question": "Redis 内存快满了，之前的持久化和淘汰策略是什么？", "relevant_topics": ["Redis"]},
    {"question": "要做数据大屏，数据库分库分表后怎么跨库查询？", "relevant_topics": ["数据库", "Redis"]},
    {"question": "容器化部署后监控方案怎么选？", "relevant_topics": ["CI/CD"]},
]


def generate_history():
    from hermes_core.consolidation import generate_consolidation_corpus
    corpus = generate_consolidation_corpus(50)
    return [{"id": r["id"], "content": r["content"], "cluster": r["cluster"]} for r in corpus]


def get_embedding(text):
    resp = requests.post(LM_STUDIO_URL, json={"model": EMBEDDING_MODEL, "input": text})
    return np.array(resp.json()["data"][0]["embedding"], dtype=np.float32)


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def prefetch_top_k(query_emb, history_embs, k=5):
    scored = [(hid, cosine_sim(query_emb, heb)) for hid, heb in history_embs]
    scored.sort(key=lambda x: -x[1])
    return scored[:k]


def random_k(history_ids, k=5, seed=42):
    rng = random.Random(seed)
    return rng.sample(history_ids, min(k, len(history_ids)))


CONTEXT_PROMPT = {
    "none": "请回答以下问题，不需要参考任何历史决策：\n{question}",
    "full": "以下是团队的 50 条历史决策记录：\n{history}\n\n请基于以上历史决策回答：\n{question}",
    "prefetch": "以下是与当前问题最相关的 5 条历史决策：\n{prefetch}\n\n请基于以上相关决策回答：\n{question}",
    "random": "以下是 5 条历史决策记录：\n{random_ctx}\n\n请基于以上记录回答：\n{question}",
}


def estimate_tokens(text):
    cn = sum(1 for c in text if '一' <= c <= '鿿')
    en = len(text) - cn
    return int(cn / 1.5 + en / 4)


def run():
    print("=" * 60)
    print("EXPERIMENT 3 v2: Context Prefetch (revised)")
    print("=" * 60)

    results_dir = Path(__file__).parent.parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    history = generate_history()
    print(f"History: {len(history)} decisions")

    print("Computing embeddings...")
    history_embs = []
    for h in history:
        emb = get_embedding(h["content"])
        history_embs.append((h["id"], emb))
    scenario_embs = [get_embedding(s["question"]) for s in SCENARIOS]
    print(f"Embeddings done: {len(history_embs)} history + {len(SCENARIOS)} scenarios")

    for s in SCENARIOS:
        s["relevant_ids"] = [h["id"] for h in history if h["cluster"] in s["relevant_topics"]]

    # GLM only
    llm = LLMBenchmark(PROVIDERS["glm"])
    print(f"\n=== {llm.config.name} ===")

    strategies = ["none", "full", "prefetch", "random"]
    strat_results = {s: [] for s in strategies}

    # Track injected tokens per strategy
    total_injected_tokens = {s: 0 for s in strategies}
    # Track embedding compute cost
    emb_compute = {"none": 0, "full": 0, "prefetch": len(history) * 512, "random": 0}

    for strategy in strategies:
        for i, scenario in enumerate(SCENARIOS):
            hist_map = {h["id"]: h for h in history}
            injected_text = ""

            if strategy == "none":
                prompt = CONTEXT_PROMPT["none"].format(question=scenario["question"])
            elif strategy == "full":
                injected_text = "\n".join(f"- [{h['cluster']}] {h['content']}" for h in history)
                prompt = CONTEXT_PROMPT["full"].format(history=injected_text, question=scenario["question"])
            elif strategy == "prefetch":
                top_k = prefetch_top_k(scenario_embs[i], history_embs, k=5)
                injected_text = "\n".join(
                    f"- [{hist_map[hid]['cluster']}] {hist_map[hid]['content']}" for hid, sim in top_k
                )
                prompt = CONTEXT_PROMPT["prefetch"].format(prefetch=injected_text, question=scenario["question"])
            elif strategy == "random":
                rand_ids = random_k([h["id"] for h in history], k=5, seed=i)
                injected_text = "\n".join(
                    f"- [{hist_map[rid]['cluster']}] {hist_map[rid]['content']}" for rid in rand_ids
                )
                prompt = CONTEXT_PROMPT["random"].format(random_ctx=injected_text, question=scenario["question"])

            response = llm(prompt, max_tokens=500)

            # Count injected tokens
            injected_tok = estimate_tokens(injected_text) if injected_text else 0
            total_injected_tokens[strategy] += injected_tok

            # Check references
            relevant_ids = set(scenario["relevant_ids"])
            referenced = sum(
                1 for rid in relevant_ids
                if hist_map[rid]["content"][:20] in response
            )

            strat_results[strategy].append({
                "scenario": i,
                "response_len": len(response),
                "injected_tokens": injected_tok,
                "referenced_relevant": referenced,
                "total_relevant": len(relevant_ids),
            })

        avg_ref = sum(r["referenced_relevant"] for r in strat_results[strategy]) / len(SCENARIOS)
        avg_inj = total_injected_tokens[strategy] / len(SCENARIOS)
        print(f"  {strategy:10s}: avg_refs={avg_ref:.1f}, avg_injected_tokens={avg_inj:.0f}")

    results = {
        "experiment": "exp3_prefetch_v2",
        "params": {"n_scenarios": len(SCENARIOS), "n_history": len(history), "top_k": 5},
        "results": {
            "glm": {
                "model": llm.config.name,
                "llm_stats": llm.stats.to_dict(),
                "strategies": strat_results,
                "total_injected_tokens": total_injected_tokens,
                "emb_compute": emb_compute,
            }
        },
    }

    out_path = results_dir / "exp3_v2_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")

    # Summary
    print("\n--- Summary ---")
    print(f"  {'Strategy':<12} {'Avg Refs':>10} {'Avg Inj Tok':>12} {'Emb Compute':>12} {'Refs/Tok':>10}")
    print("  " + "-" * 58)
    for s in strategies:
        avg_ref = sum(r["referenced_relevant"] for r in strat_results[s]) / len(SCENARIOS)
        avg_tok = total_injected_tokens[s] / len(SCENARIOS)
        efficiency = avg_ref / avg_tok if avg_tok > 0 else 0
        print(f"  {s:<12} {avg_ref:>10.1f} {avg_tok:>12.0f} {emb_compute[s]:>12} {efficiency:>10.4f}")

    return results


if __name__ == "__main__":
    run()
