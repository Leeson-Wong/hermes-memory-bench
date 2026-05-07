"""Experiment 3: Prefetch — does retrieving relevant context improve response quality?

Design:
  - 10 simulated decision scenarios, each with a question + 50 historical decisions
  - 3 strategies: no context (control) / full context (inject all) / prefetch top-5
  - LLM: GLM-5.1
  - Metric: whether response references ground-truth relevant decisions
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from hermes_core.api import get_all_providers, PROVIDERS

LM_STUDIO_URL = "http://127.0.0.1:1234/v1/embeddings"
EMBEDDING_MODEL = "text-embedding-bge-small-zh-v1.5"

# ── Scenario generation ─────────────────────────────────────────────────

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


def generate_history() -> list[dict]:
    """Generate 50 historical decisions across 5 clusters."""
    from hermes_core.consolidation import generate_consolidation_corpus
    corpus = generate_consolidation_corpus(50)
    return [{"id": r["id"], "content": r["content"], "cluster": r["cluster"]} for r in corpus]


def get_embedding(text: str) -> np.ndarray:
    resp = requests.post(LM_STUDIO_URL, json={"model": EMBEDDING_MODEL, "input": text})
    data = resp.json()
    return np.array(data["data"][0]["embedding"], dtype=np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def prefetch_top_k(query_emb: np.ndarray, history_embs: list[tuple[str, np.ndarray]], k: int = 5) -> list[tuple[str, float]]:
    scored = [(hid, cosine_sim(query_emb, heb)) for hid, heb in history_embs]
    scored.sort(key=lambda x: -x[1])
    return scored[:k]


def run():
    print("=" * 60)
    print("EXPERIMENT 3: Context Prefetch")
    print("=" * 60)

    results_dir = Path(__file__).parent.parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Generate history
    history = generate_history()
    print(f"History: {len(history)} decisions")

    # Compute embeddings
    print("Computing embeddings...")
    history_embs = []
    for h in history:
        emb = get_embedding(h["content"])
        history_embs.append((h["id"], emb))
    scenario_embs = [get_embedding(s["question"]) for s in SCENARIOS]
    print(f"Embeddings done: {len(history_embs)} history + {len(SCENARIOS)} scenarios")

    # Tag relevant history per scenario
    for s in SCENARIOS:
        s["relevant_ids"] = [h["id"] for h in history if h["cluster"] in s["relevant_topics"]]

    results = {}

    # --- Control A: No context ---
    # --- Control B: Full context (all 50 decisions) ---
    # --- Experiment: Prefetch top-5 ---

    CONTEXT_PROMPT = {
        "none": "请回答以下问题，不需要参考任何历史决策：\n{question}",
        "full": "以下是团队的 50 条历史决策记录：\n{history}\n\n请基于以上历史决策回答：\n{question}",
        "prefetch": "以下是与当前问题最相关的 5 条历史决策：\n{prefetch}\n\n请基于以上相关决策回答：\n{question}",
    }

    providers = get_all_providers()

    for provider_key, llm_bench in providers.items():
        print(f"\n=== {llm_bench.config.name} ===")
        provider_results = {"none": [], "full": [], "prefetch": []}

        for strategy in ["none", "full", "prefetch"]:
            references_found = 0
            total_relevant = 0

            for i, scenario in enumerate(SCENARIOS):
                if strategy == "none":
                    prompt = CONTEXT_PROMPT["none"].format(question=scenario["question"])
                elif strategy == "full":
                    hist_text = "\n".join(f"- [{h['cluster']}] {h['content']}" for h in history)
                    prompt = CONTEXT_PROMPT["full"].format(history=hist_text, question=scenario["question"])
                elif strategy == "prefetch":
                    top_k = prefetch_top_k(scenario_embs[i], history_embs, k=5)
                    hist_map = {h["id"]: h for h in history}
                    prefetch_text = "\n".join(f"- [{hist_map[hid]['cluster']}] {hist_map[hid]['content']}" for hid, sim in top_k)
                    prompt = CONTEXT_PROMPT["prefetch"].format(prefetch=prefetch_text, question=scenario["question"])

                response = llm_bench(prompt, max_tokens=500)

                # Check if response references relevant decisions
                relevant_ids = set(scenario["relevant_ids"])
                referenced = sum(1 for rid in relevant_ids if history[[h["id"] for h in history].index(rid)]["content"][:20] in response)
                references_found += referenced
                total_relevant += len(relevant_ids)

                provider_results[strategy].append({
                    "scenario": i,
                    "response_len": len(response),
                    "referenced_relevant": referenced,
                    "total_relevant": len(relevant_ids),
                })

                if (i + 1) % 5 == 0:
                    print(f"  {strategy}: {i+1}/{len(SCENARIOS)} done")

            # Summarize
            avg_ref = sum(r["referenced_relevant"] for r in provider_results[strategy]) / max(len(provider_results[strategy]), 1)
            print(f"  {strategy}: avg references to relevant decisions = {avg_ref:.2f}")

        results[provider_key] = {
            "model": llm_bench.config.name,
            "llm_stats": llm_bench.stats.to_dict(),
            "strategies": provider_results,
        }

    # Save
    out = {
        "experiment": "exp3_prefetch",
        "params": {"n_scenarios": len(SCENARIOS), "n_history": len(history), "top_k": 5},
        "results": results,
    }
    with open(results_dir / "exp3_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {results_dir / 'exp3_results.json'}")
    return out


if __name__ == "__main__":
    run()
