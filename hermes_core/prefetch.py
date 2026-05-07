"""Prefetch-based memory retrieval inspired by Hermes MemoryProvider lifecycle.

Standalone module: no dependency on Hermes internals. Uses numpy cosine
similarity over injectable embeddings.  Lifecycle mirrors Hermes:

  add_record()  -> seed the knowledge store
  sync_turn()   -> persist turn + extract decisions
  prefetch()    -> top-K retrieval before the next turn

Usage:
    from hermes_core.prefetch import PrefetchMemory, MockEmbedding

    mem = PrefetchMemory(MockEmbedding(), top_k=3)
    mem.add_record("r1", "We chose Redis for session caching")
    results = mem.prefetch("How do we handle sessions?")
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Embedding protocol + mock implementation
# ---------------------------------------------------------------------------

class EmbeddingFn(Protocol):
    """Callable that maps text to a dense vector."""
    def __call__(self, text: str) -> np.ndarray: ...


class MockEmbedding:
    """Deterministic embedding for testing. Uses hash-based vectors.

    Texts sharing words get similar vectors (each word hashes into the same
    seed space).  Normalised to unit length so cosine similarity is just
    the dot product.
    """
    def __init__(self, dim: int = 128):
        self.dim = dim

    def __call__(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        words = re.findall(r"[a-zA-Z0-9]+", text.lower())
        for w in words:
            h = int(hashlib.sha256(w.encode()).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


# ---------------------------------------------------------------------------
# PrefetchMemory
# ---------------------------------------------------------------------------

class PrefetchMemory:
    """Memory store with prefetch-based retrieval for AI context injection."""

    def __init__(self, embedding_fn: EmbeddingFn, top_k: int = 5):
        self._embed = embedding_fn
        self._top_k = top_k
        # Parallel arrays kept in insertion order
        self._ids: list[str] = []
        self._contents: list[str] = []
        self._metas: list[dict] = []
        self._vectors: list[np.ndarray] = []

    # -- mutation -----------------------------------------------------------

    def add_record(self, record_id: str, content: str,
                   metadata: dict | None = None) -> None:
        """Add a knowledge record and compute its embedding."""
        self._ids.append(record_id)
        self._contents.append(content)
        self._metas.append(metadata or {})
        self._vectors.append(self._embed(content))

    def sync_turn(self, user_content: str, assistant_content: str) -> None:
        """Persist a completed turn.  Extracts sentences that look like
        decisions (contain 'chose', 'decided', 'will use', etc.) and adds
        them as records so they become retrievable in future turns."""
        decision_keywords = (
            "chose", "decided", "will use", "adopted", "switched to",
            "selected", "agreed on", "standardized on", "going with",
        )
        combined = f"{user_content} {assistant_content}"
        for sentence in re.split(r"(?<=[.!?])\s+", combined):
            if any(kw in sentence.lower() for kw in decision_keywords):
                rid = f"turn-{len(self._ids)}"
                self.add_record(rid, sentence, {"source": "sync_turn"})

    # -- retrieval ----------------------------------------------------------

    def prefetch(self, query: str) -> list[dict]:
        """Retrieve top-K relevant records for the upcoming turn.

        Returns list of {record_id, content, similarity, metadata} sorted
        by descending similarity.
        """
        if not self._vectors:
            return []
        q_vec = self._embed(query)
        matrix = np.array(self._vectors)          # (N, dim)
        sims = matrix @ q_vec                     # cosine (unit vectors)
        k = min(self._top_k, len(sims))
        if k >= len(sims):
            top_idx = np.argsort(-sims)[:k]
        else:
            top_idx = np.argpartition(-sims, k)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        return [
            {
                "record_id": self._ids[i],
                "content": self._contents[i],
                "similarity": float(sims[i]),
                "metadata": self._metas[i],
            }
            for i in top_idx
        ]

    # -- inspection ---------------------------------------------------------

    def get_all_embeddings(self) -> np.ndarray:
        """Return all stored embeddings as a matrix (N x dim)."""
        if not self._vectors:
            return np.empty((0, 0), dtype=np.float32)
        return np.array(self._vectors)


# ---------------------------------------------------------------------------
# Test data generator
# ---------------------------------------------------------------------------

_CLUSTER_TEMPLATES: dict[str, list[str]] = {
    "Redis caching": [
        "We chose Redis for distributed session caching across API servers.",
        "Redis pub/sub handles cache invalidation events between nodes.",
        "Set Redis maxmemory-policy to allkeys-lru for session TTL eviction.",
        "Decided on Redis Cluster mode for horizontal scaling of the cache tier.",
        "Use Redis sorted sets for leaderboard and ranking features.",
        "Adopted Redis Streams for real-time event propagation to workers.",
        "Switched to Redis 7 functions for atomic Lua scripting of cache logic.",
        "Standardized on Redis Sentinel for high-availability failover.",
        "Chose ioredis client with cluster-aware connection pooling.",
        "Going with Redis as the rate limiter backing store with sliding window.",
    ],
    "API design": [
        "Decided on REST with OpenAPI 3.1 spec for all public-facing endpoints.",
        "API versioning strategy: URL path (/v2/) with sunset headers.",
        "Adopted JSON:API specification for consistent resource envelope format.",
        "Standardized on snake_case for request/response JSON field names.",
        "Will use cursor-based pagination for all list endpoints.",
        "Chose JWT with short TTL plus refresh token rotation for auth.",
        "Agreed on problem+json (RFC 7807) for structured error responses.",
        "Selected gRPC with protobuf for internal service-to-service calls.",
        "API rate limits: 100 req/min for free tier, 1000 for pro.",
        "Going with idempotency keys for all write endpoints.",
    ],
    "Auth system": [
        "Decided on OAuth 2.0 + PKCE flow for all client-side applications.",
        "Adopted RBAC with role hierarchy: viewer, editor, admin, owner.",
        "Chose bcrypt with cost factor 12 for password hashing.",
        "Will use short-lived access tokens (15 min) with httpOnly refresh cookies.",
        "Selected SAML 2.0 for enterprise SSO integration.",
        "Standardized on magic link authentication as passwordless fallback.",
        "Agreed on session binding to device fingerprint for anomaly detection.",
        "Switched to Argon2id for new user registrations, migrate bcrypt on login.",
        "Adopted TOTP-based 2FA with recovery codes stored encrypted.",
        "Going with centralized auth service with shared nothing architecture.",
    ],
    "Database schema": [
        "Chose PostgreSQL with UUIDv7 primary keys for all new tables.",
        "Decided on soft-delete pattern with deleted_at timestamp column.",
        "Adopted event-sourcing for the orders aggregate with snapshot table.",
        "Standardized on JSONB columns for extensible metadata storage.",
        "Will use row-level security policies for multi-tenant data isolation.",
        "Agreed on database-level audit triggers for all mutation operations.",
        "Selected optimistic concurrency control using version columns.",
        "Database migrations managed by Prisma with shadow database for diffs.",
        "Switched to partitioned tables by month for high-volume event logs.",
        "Going with read replicas for reporting queries to offload primary.",
    ],
    "CI/CD pipeline": [
        "Decided on GitHub Actions with reusable workflow templates.",
        "Adopted trunk-based development with feature flags for gradual rollout.",
        "Chose Docker multi-stage builds for minimal production images.",
        "Will use semantic versioning with conventional commits for releases.",
        "Selected ArgoCD for GitOps-based Kubernetes deployments.",
        "Standardized on branch protection rules requiring 2 approvals.",
        "Agreed on canary deployments with automatic rollback on error spike.",
        "Switched to Turborepo for monorepo build caching and task pipeline.",
        "Adopted dependency scanning and SAST in every pull request.",
        "Going with Terraform modules for all infrastructure provisioning.",
    ],
}


def generate_decision_corpus(n: int = 200) -> list[dict]:
    """Generate *n* simulated decision records with known clusters.

    Each record dict has keys: id, content, cluster, metadata.
    Records within a cluster share overlapping vocabulary so mock
    embeddings group them together.
    """
    import random
    rng = random.Random(42)
    clusters = list(_CLUSTER_TEMPLATES.keys())
    per_cluster = n // len(clusters)
    remainder = n - per_cluster * len(clusters)
    records: list[dict] = []
    counter = 0
    for ci, cluster in enumerate(clusters):
        count = per_cluster + (1 if ci < remainder else 0)
        for _ in range(count):
            template = rng.choice(_CLUSTER_TEMPLATES[cluster])
            # Vary the wording slightly while keeping key terms
            suffix = rng.choice([
                "", f" (confirmed {2024 + rng.randint(0, 2)})",
                f" — team consensus.", f" — approved by tech lead.",
            ])
            records.append({
                "id": f"rec-{counter:04d}",
                "content": template + suffix,
                "cluster": cluster,
                "metadata": {"cluster": cluster},
            })
            counter += 1
    rng.shuffle(records)
    return records
