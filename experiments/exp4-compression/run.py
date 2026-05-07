"""Experiment 4: Context Compression — 3-phase vs simple truncation.

Compares:
  - Strategy: 3-phase compression vs simple truncation
  - LLM: GLM-5.1
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from hermes_core.api import get_all_providers, PROVIDERS
from hermes_core.compressor import (
    ContextCompressor,
    MockLLM,
    generate_long_conversation,
    count_tokens,
)


def simple_truncate(messages: list[dict], budget_tokens: int) -> tuple[list[dict], dict]:
    """Control A: keep head (system) + tail, cut middle."""
    total = sum(count_tokens(m["content"]) for m in messages)
    if total <= budget_tokens:
        return messages, {"compression_ratio": 1.0, "method": "truncation", "phases": {}}

    # Keep first message (system) and as many tail messages as fit
    head = [messages[0]]
    head_tokens = count_tokens(messages[0]["content"])
    remaining = budget_tokens - head_tokens

    tail = []
    tail_tokens = 0
    for m in reversed(messages[1:]):
        t = count_tokens(m["content"])
        if tail_tokens + t > remaining:
            break
        tail.insert(0, m)
        tail_tokens += t

    result = head + tail
    new_total = sum(count_tokens(m["content"]) for m in result)
    return result, {
        "original_tokens": total,
        "compressed_tokens": new_total,
        "compression_ratio": round(new_total / total, 4),
        "method": "truncation",
        "pruned_count": len(messages) - len(result),
    }


def run():
    print("=" * 60)
    print("EXPERIMENT 4: Context Compression")
    print("=" * 60)

    results_dir = Path(__file__).parent.parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Generate test conversations
    conversations = []
    for i in range(5):
        conv = generate_long_conversation(n_turns=80, tokens_per_turn=300)
        total = sum(count_tokens(m["content"]) for m in conv)
        conversations.append({"id": f"conv_{i}", "messages": conv, "total_tokens": total})

    avg_tokens = sum(c["total_tokens"] for c in conversations) / len(conversations)
    budget = 8000
    print(f"Generated {len(conversations)} conversations, avg {avg_tokens:.0f} tokens")
    print(f"Budget: {budget} tokens")

    results = {}

    # --- Control: simple truncation ---
    print("\n--- Control: Simple truncation ---")
    trunc_ratios = []
    for conv in conversations:
        _, metrics = simple_truncate(conv["messages"], budget)
        trunc_ratios.append(metrics["compression_ratio"])
    avg_trunc = sum(trunc_ratios) / len(trunc_ratios)
    print(f"  Avg compression ratio: {avg_trunc:.4f}")
    results["truncation"] = {"avg_ratio": round(avg_trunc, 4), "per_conv": trunc_ratios}

    # --- Experiment: 3-phase compression (per provider) ---
    providers = get_all_providers()

    for provider_key, llm_bench in providers.items():
        print(f"\n--- 3-phase compression: {llm_bench.config.name} ---")
        ratios = []
        phase_data = []
        for conv in conversations:
            comp = ContextCompressor(
                llm_fn=llm_bench,
                context_window=int(conv["total_tokens"] * 1.2),
                threshold_percent=0.5,
            )
            compressed, metrics = comp.compress(conv["messages"])
            ratios.append(metrics["compression_ratio"])
            phase_data.append(metrics)

        avg_ratio = sum(ratios) / len(ratios)
        print(f"  Avg compression ratio: {avg_ratio:.4f}")
        print(f"  LLM stats: {llm_bench.stats.to_dict()}")

        results[f"3phase_{provider_key}"] = {
            "model": llm_bench.config.name,
            "avg_ratio": round(avg_ratio, 4),
            "per_conv": [round(r, 4) for r in ratios],
            "llm_stats": llm_bench.stats.to_dict(),
        }

    # Save
    out = {
        "experiment": "exp4_compression",
        "params": {"n_conversations": len(conversations), "budget_tokens": budget, "avg_original_tokens": round(avg_tokens)},
        "results": results,
    }
    with open(results_dir / "exp4_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n{'Strategy':<25} {'Avg Ratio':>10} {'Target':>10}")
    print("-" * 45)
    print(f"{'Target ratio':.<25} {'0.20':>10} {'20%':>10}")
    for key, r in results.items():
        if "avg_ratio" in r:
            print(f"{key:<25} {r['avg_ratio']:>10.4f} {'':>10}")

    print(f"\nResults saved to {results_dir / 'exp4_results.json'}")
    return out


if __name__ == "__main__":
    run()
