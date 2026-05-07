"""Consolidation engine — LLM-driven umbrella-building knowledge merge.

Ported from Hermes agent/curator.py consolidation logic.
Tests whether LLM-based merging outperforms pure embedding similarity.
"""

import json
import re
from typing import Optional

import numpy as np


# ── Structured merge prompt (from Hermes curator) ──────────────────────

MERGE_PROMPT = """You are a knowledge curator. Given two knowledge records, determine if they should be merged under a common umbrella concept.

Record A: {record_a}
Record B: {record_b}

Answer in YAML format:
```yaml
should_merge: true/false
umbrella_name: "short descriptive name if merging, empty otherwise"
reason: "one sentence explaining why or why not"
```
"""


# ── Core engine ─────────────────────────────────────────────────────────

class ConsolidationEngine:
    """Evaluate knowledge merge candidates using embedding + LLM."""

    def __init__(self, embedding_fn, llm_fn, similarity_threshold: float = 0.60):
        """
        Args:
            embedding_fn: callable(text) -> np.ndarray
            llm_fn: callable(prompt, max_tokens) -> str
            similarity_threshold: minimum cosine similarity to consider as candidate pair
        """
        self.embedding_fn = embedding_fn
        self.llm_fn = llm_fn
        self.similarity_threshold = similarity_threshold
        self.records = {}  # id -> {content, embedding, metadata}
        self._call_count = 0
        self._total_tokens = 0

    def add_record(self, record_id: str, content: str, metadata: dict | None = None):
        emb = self.embedding_fn(content)
        self.records[record_id] = {
            "content": content,
            "embedding": emb,
            "metadata": metadata or {},
        }

    @staticmethod
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def find_candidates(self) -> list[tuple[str, str, float]]:
        """Find all pairs with similarity above threshold. Returns [(id_a, id_b, sim)]."""
        ids = list(self.records.keys())
        candidates = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                sim = self.cosine_sim(
                    self.records[ids[i]]["embedding"],
                    self.records[ids[j]]["embedding"],
                )
                if sim >= self.similarity_threshold:
                    candidates.append((ids[i], ids[j], sim))
        candidates.sort(key=lambda x: -x[2])
        return candidates

    def judge_pair(self, id_a: str, id_b: str) -> dict:
        """Ask LLM whether two records should be merged."""
        rec_a = self.records[id_a]["content"]
        rec_b = self.records[id_b]["content"]
        prompt = MERGE_PROMPT.format(record_a=rec_a, record_b=rec_b)
        response = self.llm_fn(prompt, max_tokens=200)
        self._call_count += 1
        self._total_tokens += len(prompt) // 4 + len(response) // 4

        # Parse YAML response
        result = {"should_merge": False, "umbrella_name": "", "reason": "", "raw": response}
        try:
            yaml_match = re.search(r"```yaml\s*(.*?)```", response, re.DOTALL)
            text = yaml_match.group(1) if yaml_match else response
            for line in text.strip().split("\n"):
                line = line.strip()
                if line.startswith("should_merge:"):
                    result["should_merge"] = "true" in line.lower()
                elif line.startswith("umbrella_name:"):
                    result["umbrella_name"] = line.split(":", 1)[1].strip().strip('"').strip("'")
                elif line.startswith("reason:"):
                    result["reason"] = line.split(":", 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass
        return result

    def judge_all_candidates(self) -> list[dict]:
        """Run LLM judgment on all candidate pairs."""
        candidates = self.find_candidates()
        results = []
        for id_a, id_b, sim in candidates:
            judgment = self.judge_pair(id_a, id_b)
            results.append({
                "id_a": id_a,
                "id_b": id_b,
                "similarity": round(sim, 4),
                **judgment,
            })
        return results

    def pure_embedding_merge(self, merge_threshold: float = 0.85) -> list[dict]:
        """Control group: merge purely based on embedding similarity, no LLM."""
        results = []
        ids = list(self.records.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                sim = self.cosine_sim(
                    self.records[ids[i]]["embedding"],
                    self.records[ids[j]]["embedding"],
                )
                if sim >= merge_threshold:
                    results.append({
                        "id_a": ids[i],
                        "id_b": ids[j],
                        "similarity": round(sim, 4),
                        "should_merge": True,
                        "method": "embedding_only",
                    })
        return results

    @property
    def stats(self) -> dict:
        return {"llm_calls": self._call_count, "est_tokens": self._total_tokens}


# ── Data generators ─────────────────────────────────────────────────────

def generate_consolidation_corpus(n: int = 200, n_clusters: int = 5) -> list[dict]:
    """Generate n records in n_clusters topic clusters with known merge targets."""
    clusters = [
        {
            "name": "Redis",
            "records": [
                "使用 Redis 做缓存策略，设置了 30 分钟 TTL，采用 LRU 淘汰",
                "Redis 缓存击穿问题，使用互斥锁方案解决",
                "Redis 集群方案选择：Codis vs Redis Cluster，选了 Cluster",
                "Redis 持久化策略：RDB + AOF 混合持久化",
                "Redis 作为消息队列使用，替代 RabbitMQ 处理简单场景",
                "Redis 分布式锁实现，使用 Redlock 算法",
                "Redis 内存优化：使用 Hash 结构替代 String 存储对象",
                "Redis 主从同步延迟问题排查",
                "Redis Sentinel 哨兵模式配置与故障转移",
                "Redis Pipeline 批量操作优化网络延迟",
            ],
            "should_merge_with_self": True,
        },
        {
            "name": "API 设计",
            "records": [
                "RESTful API 设计规范：统一返回格式 {code, data, message}",
                "API 版本管理策略：URL path versioning /v1/ /v2/",
                "API 限流方案：令牌桶算法，单用户 100 req/min",
                "GraphQL vs REST 对比，团队选择 REST 保持简单",
                "API 网关选型：Kong vs 自研，选了 Kong",
                "API 鉴权方案：JWT + Refresh Token 双 token 方案",
                "API 文档自动生成：Swagger/OpenAPI 3.0",
                "API 幂等性设计：使用 request_id 去重",
                "gRPC 用于内部服务间通信，REST 用于对外",
                "API 错误码设计：5 位数字编码，前 2 位模块后 3 位错误",
            ],
            "should_merge_with_self": True,
        },
        {
            "name": "认证系统",
            "records": [
                "用户认证从 Session 迁移到 JWT",
                "OAuth2.0 授权码模式接入第三方登录",
                "RBAC 权限模型设计：用户-角色-权限三层",
                "密码存储方案：bcrypt hash + salt",
                "SSO 单点登录实现：基于 CAS 协议",
                "多因子认证 MFA：TOTP 方案选择",
                "权限缓存策略：用户权限变更后 5 分钟生效",
                "账号锁定策略：连续 5 次失败锁定 30 分钟",
                "Token 安全：HttpOnly + Secure Cookie 存储",
                "LDAP 对接企业内部账号体系",
            ],
            "should_merge_with_self": True,
        },
        {
            "name": "数据库",
            "records": [
                "MySQL 分库分表方案：ShardingSphere 中间件",
                "PostgreSQL JSONB 字段存储灵活配置",
                "数据库连接池配置：HikariCP maxPoolSize=20",
                "慢查询优化：联合索引覆盖扫描代替回表",
                "读写分离方案：ProxySQL 路由读写请求",
                "数据迁移方案：双写 + 灰度切换",
                "数据库备份策略：每日全量 + 实时 binlog",
                "MySQL 8.0 升级注意事项：窗口函数和 CTE",
                "MongoDB 用于日志存储，设置 TTL 自动过期",
                "数据库监控：Prometheus + Grafana 看板",
            ],
            "should_merge_with_self": True,
        },
        {
            "name": "CI/CD",
            "records": [
                "Jenkins Pipeline 流水线改造：声明式语法",
                "Docker 多阶段构建优化镜像大小",
                "Kubernetes 部署策略：滚动更新 + 就绪探针",
                "Helm Chart 模板化部署配置",
                "自动化测试集成：单元测试覆盖率门控 80%",
                "灰度发布方案：Canary Deployment + Nginx 流量分割",
                "制品管理：Harbor 私有镜像仓库",
                "环境管理：dev/staging/prod 三环境隔离",
                "监控告警：Prometheus AlertManager 规则配置",
                "日志收集：EFK (Elasticsearch + Fluentd + Kibana)",
            ],
            "should_merge_with_self": True,
        },
    ]

    records = []
    for cluster in clusters:
        for content in cluster["records"]:
            records.append({
                "id": f"{cluster['name']}_{len(records)}",
                "content": content,
                "cluster": cluster["name"],
                "should_merge_with_cluster": cluster["should_merge_with_self"],
            })

    # Pad to n if needed
    while len(records) < n:
        import random
        cluster = random.choice(clusters)
        records.append({
            "id": f"{cluster['name']}_{len(records)}",
            "content": f"补充记录：{cluster['name']}相关决策 #{len(records)}",
            "cluster": cluster["name"],
            "should_merge_with_cluster": True,
        })

    return records[:n]


def build_ground_truth(records: list[dict]) -> list[dict]:
    """Build ground truth merge pairs: same cluster = should merge."""
    pairs = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            if records[i]["cluster"] == records[j]["cluster"]:
                pairs.append({
                    "id_a": records[i]["id"],
                    "id_b": records[j]["id"],
                    "should_merge": True,
                    "cluster": records[i]["cluster"],
                })
    return pairs
