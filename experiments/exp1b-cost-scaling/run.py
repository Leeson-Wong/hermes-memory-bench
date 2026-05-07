"""Experiment 1b: Cost Scaling — lifecycle management cost-benefit analysis.

Key question: Does lifecycle management save money at scale?

What we measure:
  1. Per-query cost over time (does it converge or grow?)
  2. Cumulative cost divergence (total spend over lifetime)
  3. Recall of useful records (quality retention)
  4. Signal-to-noise ratio (pool quality)

Two simulation modes:
  A. Fixed dataset: N records at day 0, multiple volumes (isolates scaling)
  B. Continuous growth: 5 new records/day over 730 days (realistic SaaS usage)

Cost model per query:
  - embedding scan: pool_size * dim (cosine similarity computation)
  - token injection: top_K * avg_record_tokens (context added to prompt)
  - Per-query cost = embedding scan + token injection

Quality model:
  - Recall: fraction of useful records still searchable
  - Signal ratio: useful / pool_size (higher = less noise)
  - Useful = high-freq (top 10%) + mid-freq (next 20%) records

No LLM or embedding API needed - pure analytical simulation.

Usage:
    cd F:/mime/learn
    python -m experiments.exp1b-cost-scaling.run
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DATA_VOLUMES = [200, 500, 1000, 2000, 5000]
SIM_DAYS_FIXED = 180
GROWTH_DAYS = 730  # 2 years
QUERIES_PER_DAY = 10
TOP_K = 5
AVG_RECORD_TOKENS = 50
EMBEDDING_DIM = 512

# ── Lifecycle managers ─────────────────────────────────────────────────────

class HermesLifecycle:
    def __init__(self, stale_days=30, archive_days=90):
        self.records = {}
        self.stale_days = stale_days
        self.archive_days = archive_days

    def add(self, rid, created_at, pinned=False, freq_tier="low"):
        self.records[rid] = {
            "state": "active", "created_at": created_at,
            "last_ref": None, "pinned": pinned, "freq_tier": freq_tier,
        }

    def touch(self, rid, at):
        if rid in self.records:
            self.records[rid]["last_ref"] = at

    def tick(self, now):
        now_dt = _parse(now)
        stale_cut = now_dt - timedelta(days=self.stale_days)
        archive_cut = now_dt - timedelta(days=self.archive_days)
        counts = {"stale": 0, "archived": 0, "reactivated": 0}
        for r in self.records.values():
            if r["pinned"]:
                continue
            anchor = _parse(r["last_ref"] or r["created_at"])
            if anchor <= archive_cut and r["state"] != "archived":
                r["state"] = "archived"; counts["archived"] += 1
            elif anchor <= stale_cut and r["state"] == "active":
                r["state"] = "stale"; counts["stale"] += 1
            elif anchor > stale_cut and r["state"] == "stale":
                r["state"] = "active"; counts["reactivated"] += 1
        return counts

    def searchable_pool(self):
        return {rid for rid, r in self.records.items() if r["state"] == "active"}

    def total_stored(self):
        return len(self.records)

    def pool_by_tier(self, pool):
        tiers = {"high": 0, "mid": 0, "low": 0}
        for rid in pool:
            t = self.records[rid].get("freq_tier", "low")
            tiers[t] = tiers.get(t, 0) + 1
        return tiers


class SimpleExpiry:
    def __init__(self, expire_days=30):
        self.records = {}
        self.expire_days = expire_days

    def add(self, rid, created_at, freq_tier="low"):
        self.records[rid] = {
            "created_at": created_at, "last_ref": None,
            "expired": False, "freq_tier": freq_tier,
        }

    def touch(self, rid, at):
        if rid in self.records:
            self.records[rid]["last_ref"] = at

    def tick(self, now):
        cut = _parse(now) - timedelta(days=self.expire_days)
        expired = 0
        for r in self.records.values():
            if r["expired"]:
                continue
            anchor = _parse(r["last_ref"] or r["created_at"])
            if anchor < cut:
                r["expired"] = True; expired += 1
        return {"expired": expired}

    def searchable_pool(self):
        return {rid for rid, r in self.records.items() if not r["expired"]}

    def total_stored(self):
        return len(self.records)


class NoManagement:
    def __init__(self):
        self.records = {}

    def add(self, rid, created_at, freq_tier="low"):
        self.records[rid] = {"freq_tier": freq_tier}

    def touch(self, rid, at):
        pass

    def tick(self, now):
        return {}

    def searchable_pool(self):
        return set(self.records.keys())

    def total_stored(self):
        return len(self.records)


# ── Helpers ────────────────────────────────────────────────────────────────

def _parse(s):
    dt = datetime.fromisoformat(s)
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def generate_records(n, days_span, seed=42):
    rng = random.Random(seed)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    high_count = max(1, n // 10)
    mid_count = max(1, n // 5)
    records = []

    for i in range(n):
        created = base + timedelta(days=rng.uniform(0, days_span * 0.3))
        if i < high_count:
            tier, gap_range = "high", (1, 5)
        elif i < high_count + mid_count:
            tier, gap_range = "mid", (10, 30)
        else:
            tier, gap_range = "low", (60, 180)
        pinned = rng.random() < 0.02
        refs = []
        t = created
        end = base + timedelta(days=days_span)
        while t < end:
            t += timedelta(days=rng.randint(*gap_range))
            if t < end:
                refs.append(t.strftime("%Y-%m-%d"))
        records.append({
            "record_id": f"rec-{i:04d}",
            "created_at": created.isoformat(),
            "pinned": pinned,
            "references": refs,
            "freq_tier": tier,
        })
    return records


def build_ref_schedule(records):
    schedule = {}
    for r in records:
        for d in r["references"]:
            schedule.setdefault(d, []).append(r["record_id"])
    return schedule


# ── Cost simulation ────────────────────────────────────────────────────────

def simulate_fixed(manager, records, ref_schedule, days, qpd):
    """Fixed dataset simulation with per-tier recall tracking."""
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    tiers = {"high": set(), "mid": set(), "low": set()}
    for r in records:
        tiers[r["freq_tier"]].add(r["record_id"])

    snapshots = []
    cum_ops = 0
    cum_tokens = 0

    for day in range(days):
        now = (base + timedelta(days=day)).isoformat()
        date_str = now[:10]

        if date_str in ref_schedule:
            for rid in ref_schedule[date_str]:
                manager.touch(rid, now)
        manager.tick(now)

        pool = manager.searchable_pool()
        pool_size = len(pool)
        day_ops = pool_size * EMBEDDING_DIM * qpd
        day_tokens = min(TOP_K, pool_size) * AVG_RECORD_TOKENS * qpd
        cum_ops += day_ops
        cum_tokens += day_tokens

        # Per-tier recall
        recall = {}
        for tier, ids in tiers.items():
            if ids:
                recall[tier] = round(len(ids & pool) / len(ids), 4)
            else:
                recall[tier] = 0

        # Signal ratio: high+mid in pool / pool size
        useful = len((tiers["high"] | tiers["mid"]) & pool)
        signal_ratio = round(useful / pool_size, 4) if pool_size else 0

        # Per-query cost
        per_query_ops = pool_size * EMBEDDING_DIM
        per_query_tokens = min(TOP_K, pool_size) * AVG_RECORD_TOKENS

        snapshots.append({
            "day": day,
            "pool_size": pool_size,
            "total_stored": manager.total_stored(),
            "per_query_ops": per_query_ops,
            "per_query_tokens": per_query_tokens,
            "cumulative_ops": cum_ops,
            "cumulative_tokens": cum_tokens,
            "recall_high": recall["high"],
            "recall_mid": recall["mid"],
            "recall_low": recall["low"],
            "signal_ratio": signal_ratio,
        })

    return snapshots


def simulate_growing(cls, records_per_day=5, days=730, qpd=10, **kwargs):
    """Continuous growth simulation — the key experiment.

    Realistic model: team creates decisions daily. Early decisions are
    "useful" (foundational knowledge). Later decisions reference them.
    """
    rng = random.Random(99)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    manager = cls(**kwargs) if kwargs else cls()

    useful_ids = set()  # records from first 30 days = foundational knowledge
    counter = 0

    snapshots = []
    cum_ops = 0
    cum_tokens = 0

    for day in range(days):
        now = (base + timedelta(days=day)).isoformat()

        # Add new records
        for _ in range(records_per_day):
            rid = f"rec-{counter:04d}"
            tier = "high" if day < 30 else "mid" if day < 90 else "low"
            if isinstance(manager, HermesLifecycle):
                manager.add(rid, now, pinned=(rng.random() < 0.02), freq_tier=tier)
            else:
                manager.add(rid, now, freq_tier=tier)
            if day < 30:
                useful_ids.add(rid)
            counter += 1

        # Reference useful records (foundational knowledge gets cited)
        for uid in list(useful_ids):
            if rng.random() < 0.04:  # 4% per useful record per day
                manager.touch(uid, now)

        # Also reference recent records (people reference recent decisions)
        if day > 7:
            recent_ref_count = rng.randint(1, 3)
            all_rids = list(manager.records.keys())
            for _ in range(recent_ref_count):
                recent_start = max(0, len(all_rids) - records_per_day * 7)
                rid = all_rids[rng.randint(recent_start, len(all_rids) - 1)]
                manager.touch(rid, now)

        manager.tick(now)

        pool = manager.searchable_pool()
        pool_size = len(pool)

        day_ops = pool_size * EMBEDDING_DIM * qpd
        day_tokens = min(TOP_K, pool_size) * AVG_RECORD_TOKENS * qpd
        cum_ops += day_ops
        cum_tokens += day_tokens

        useful_in_pool = useful_ids & pool
        recall = round(len(useful_in_pool) / len(useful_ids), 4) if useful_ids else 0

        per_query_ops = pool_size * EMBEDDING_DIM
        per_query_tokens = min(TOP_K, pool_size) * AVG_RECORD_TOKENS

        snapshots.append({
            "day": day,
            "pool_size": pool_size,
            "total_stored": manager.total_stored(),
            "per_query_ops": per_query_ops,
            "per_query_tokens": per_query_tokens,
            "cumulative_ops": cum_ops,
            "cumulative_tokens": cum_tokens,
            "useful_in_pool": len(useful_in_pool),
            "total_useful": len(useful_ids),
            "recall": recall,
            "cost_ratio_vs_noop": None,  # filled later
        })

    return snapshots


# ── Main runner ────────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("EXPERIMENT 1b: Cost Scaling Analysis")
    print("=" * 60)

    all_results = {}

    # ── Part A: Fixed dataset at multiple volumes ──────────────────────────

    print("\n--- Part A: Fixed dataset scaling ---")
    scaling = {}

    for n in DATA_VOLUMES:
        print(f"\n  Volume: {n} records")
        records = generate_records(n, SIM_DAYS_FIXED)
        ref_schedule = build_ref_schedule(records)
        tiers = {"high": sum(1 for r in records if r["freq_tier"] == "high"),
                 "mid": sum(1 for r in records if r["freq_tier"] == "mid"),
                 "low": sum(1 for r in records if r["freq_tier"] == "low")}
        print(f"    Tiers: high={tiers['high']}, mid={tiers['mid']}, low={tiers['low']}")

        volume_data = {}
        for name, create in [
            ("hermes", lambda: HermesLifecycle(stale_days=30, archive_days=90)),
            ("simple", lambda: SimpleExpiry(expire_days=30)),
            ("noop", lambda: NoManagement()),
        ]:
            mgr = create()
            for r in records:
                if isinstance(mgr, HermesLifecycle):
                    mgr.add(r["record_id"], r["created_at"], r["pinned"], r["freq_tier"])
                else:
                    mgr.add(r["record_id"], r["created_at"], r["freq_tier"])

            snaps = simulate_fixed(mgr, records, ref_schedule, SIM_DAYS_FIXED, QUERIES_PER_DAY)
            volume_data[name] = snaps
            f = snaps[-1]
            print(f"    {name:8s}: pool={f['pool_size']:>5}, "
                  f"recall_h={f['recall_high']:.0%} m={f['recall_mid']:.0%} l={f['recall_low']:.0%}, "
                  f"signal={f['signal_ratio']:.2f}, "
                  f"per_query_ops={f['per_query_ops']:>10,}")

        scaling[n] = volume_data

    all_results["fixed_scaling"] = scaling

    # ── Part B: Continuous growth over 730 days ────────────────────────────

    print(f"\n--- Part B: Continuous growth (5 records/day, {GROWTH_DAYS} days) ---")
    growth = {}

    for name, cls, kw in [
        ("hermes", HermesLifecycle, {"stale_days": 30, "archive_days": 90}),
        ("simple", SimpleExpiry, {"expire_days": 30}),
        ("noop", NoManagement, {}),
    ]:
        snaps = simulate_growing(cls, records_per_day=5, days=GROWTH_DAYS, **kw)
        growth[name] = snaps

    # Compute cross-strategy cost ratios
    for name in ["hermes", "simple"]:
        for i, snap in enumerate(growth[name]):
            noop_ops = growth["noop"][i]["per_query_ops"]
            snap["cost_ratio_vs_noop"] = round(snap["per_query_ops"] / noop_ops, 4) if noop_ops else 0

    # Print milestones
    milestones = [29, 89, 179, 364, 549, 729]
    print(f"\n  {'Strategy':<10}", end="")
    for d in milestones:
        print(f" {'Day '+str(d+1):>12}", end="")
    print()
    print("  " + "-" * (10 + 13 * len(milestones)))

    for name in ["hermes", "simple", "noop"]:
        print(f"  {name:<10}", end="")
        for d in milestones:
            s = growth[name][d]
            print(f" {s['pool_size']:>6}/{s['recall']:>5.0%}", end="")
        print()

    print(f"\n  Per-query cost (ops) over time:")
    print(f"  {'Strategy':<10}", end="")
    for d in milestones:
        print(f" {'Day '+str(d+1):>12}", end="")
    print()
    print("  " + "-" * (10 + 13 * len(milestones)))
    for name in ["hermes", "simple", "noop"]:
        print(f"  {name:<10}", end="")
        for d in milestones:
            ops = growth[name][d]["per_query_ops"]
            print(f" {ops:>12,}", end="")
        print()

    print(f"\n  Cumulative cost over time:")
    print(f"  {'Strategy':<10}", end="")
    for d in milestones:
        print(f" {'Day '+str(d+1):>14}", end="")
    print()
    print("  " + "-" * (10 + 15 * len(milestones)))
    for name in ["hermes", "simple", "noop"]:
        print(f"  {name:<10}", end="")
        for d in milestones:
            cum = growth[name][d]["cumulative_ops"]
            print(f" {cum:>14,}", end="")
        print()

    all_results["continuous_growth"] = growth

    # ── Part C: Divergence analysis ────────────────────────────────────────

    print("\n--- Cost Divergence: cumulative NoOp / Hermes ---")
    for d in milestones:
        noop_cum = growth["noop"][d]["cumulative_ops"]
        hermes_cum = growth["hermes"][d]["cumulative_ops"]
        simple_cum = growth["simple"][d]["cumulative_ops"]
        noop_recall = growth["noop"][d]["recall"]
        hermes_recall = growth["hermes"][d]["recall"]
        simple_recall = growth["simple"][d]["recall"]
        print(f"  Day {d+1:>4}: NoOp/Hermes={noop_cum/hermes_cum:.2f}x, "
              f"NoOp/Simple={noop_cum/simple_cum:.2f}x | "
              f"recall: Hermes={hermes_recall:.0%}, Simple={simple_recall:.0%}, NoOp={noop_recall:.0%}")

    # ── Save ───────────────────────────────────────────────────────────────

    # Thin data: keep every 5th day + milestones + last
    keep_days = set(milestones)
    for d in [29, 89, 179, 364, 549, 729]:
        keep_days.add(d)

    def thin_list(lst):
        out = []
        for i, item in enumerate(lst):
            if i % 10 == 0 or i in keep_days or i == len(lst) - 1:
                out.append(item)
        return out

    def thin(data):
        if isinstance(data, list):
            return thin_list(data)
        if isinstance(data, dict):
            return {k: thin(v) for k, v in data.items()}
        return data

    output = {
        "experiment": "exp1b_cost_scaling",
        "params": {
            "data_volumes": DATA_VOLUMES,
            "sim_days_fixed": SIM_DAYS_FIXED,
            "growth_days": GROWTH_DAYS,
            "queries_per_day": QUERIES_PER_DAY,
            "top_k": TOP_K,
            "avg_record_tokens": AVG_RECORD_TOKENS,
            "embedding_dim": EMBEDDING_DIM,
        },
        "results": thin(all_results),
    }

    out_path = RESULTS_DIR / "exp1b_cost_scaling.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")

    # Summary
    summary = build_summary(scaling, growth, milestones)
    summary_path = RESULTS_DIR / "exp1b_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to {summary_path}")

    return output


def build_summary(scaling, growth, milestones):
    summary = {"fixed_scaling": {}, "growth_milestones": {}, "cost_divergence": {}, "key_finding": ""}

    # Fixed scaling final
    for n, vd in scaling.items():
        summary["fixed_scaling"][str(n)] = {}
        for name in ["hermes", "simple", "noop"]:
            f = vd[name][-1]
            summary["fixed_scaling"][str(n)][name] = {
                "pool_size": f["pool_size"],
                "per_query_ops": f["per_query_ops"],
                "recall_high": f["recall_high"],
                "recall_mid": f["recall_mid"],
                "recall_low": f["recall_low"],
                "signal_ratio": f["signal_ratio"],
                "cumulative_ops": f["cumulative_ops"],
            }

    # Growth milestones
    for name in ["hermes", "simple", "noop"]:
        summary["growth_milestones"][name] = {
            f"day{d+1}": growth[name][d] for d in milestones
        }

    # Cost divergence at key points
    for d in [89, 179, 364, 549, 729]:
        noop = growth["noop"][d]["cumulative_ops"]
        hermes = growth["hermes"][d]["cumulative_ops"]
        summary["cost_divergence"][f"day{d+1}"] = {
            "noop_cumulative_ops": noop,
            "hermes_cumulative_ops": hermes,
            "divergence_ratio": round(noop / hermes, 2),
            "hermes_recall": growth["hermes"][d]["recall"],
            "noop_recall": growth["noop"][d]["recall"],
            "hermes_per_query": growth["hermes"][d]["per_query_ops"],
            "noop_per_query": growth["noop"][d]["per_query_ops"],
            "per_query_ratio": round(growth["noop"][d]["per_query_ops"] / growth["hermes"][d]["per_query_ops"], 2),
        }

    # Key finding
    d729 = summary["cost_divergence"]["day730"]
    summary["key_finding"] = (
        f"After 2 years (730 days, 5 records/day): "
        f"NoOp total cost is {d729['divergence_ratio']}x Hermes. "
        f"Per-query: NoOp costs {d729['per_query_ratio']}x more. "
        f"Hermes recall={d729['hermes_recall']:.0%}, NoOp recall={d729['noop_recall']:.0%}. "
        f"Hermes converges, NoOp grows without bound."
    )

    return summary


if __name__ == "__main__":
    run()
