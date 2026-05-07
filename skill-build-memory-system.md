# Skill: 构建类似 Hermes 的 Agent 记忆系统

> 基于工程化实验验证的 Hermes 记忆机制落地指南。每一步都有数据支撑。

---

## 概述

本 Skill 指导构建一个有生命周期的 Agent 记忆系统，核心组件：
1. **知识生命周期管理**（必做，零 LLM 成本）
2. **上下文预取**（必做，token 效率 8x 提升）
3. **上下文压缩**（长会话场景，7 步后回本）
4. **知识合并**（暂不建议，需要更好的 embedding）

实验验证：`F:\mime\learn\results\REPORT.md`
落地指南：`docs/references/hermes-memory-validation.md`

---

## Step 1: 知识生命周期状态机

### 业务价值
- **防止知识库膨胀**：无管理方案 730 天后每次检索成本增长 13.6 倍
- **保持信号质量**：信号比 0.90（无管理仅 0.30，70% 是噪声）
- **自动保留活跃知识**：reactivation 机制确保被引用的记录不会丢失

### 数据证明
730 天连续增长模拟（5 条/天，10 次查询/天）：
- Hermes 池稳定在 ~270，成本收敛
- 无管理池增长到 3,650，累计成本 7.39 倍

### 实现模式

```typescript
// 三态状态机：active → stale → archived，支持 reactivation
interface LifecycleRecord {
  id: string
  content: string
  status: 'active' | 'stale' | 'archived'
  pinned: boolean
  createdAt: Date
  lastReferencedAt: Date | null
}

function tick(record: LifecycleRecord, now: Date, config: LifecycleConfig): LifecycleRecord {
  if (record.pinned) return record
  const anchor = record.lastReferencedAt ?? record.createdAt
  const staleCutoff = daysAgo(now, config.staleDays)   // 建议 60 天
  const archiveCutoff = daysAgo(now, config.archiveDays) // 建议 180 天

  if (anchor <= archiveCutoff && record.status !== 'archived')
    return { ...record, status: 'archived' }
  if (anchor <= staleCutoff && record.status === 'active')
    return { ...record, status: 'stale' }
  if (anchor > staleCutoff && record.status === 'stale')
    return { ...record, status: 'active' }  // reactivation!
  return record
}
```

### 观测指标
- `active_ratio`：目标 40-60%
- `reactivation_count`：应该 > 0（证明机制生效）
- `searchable_pool_size`：应该趋于稳定（证明收敛）
- `per_query_embedding_ops`：应该趋于恒定

### 关键配置
| 参数 | 建议值 | 依据 |
|------|--------|------|
| stale_days | 60 | Decision.ai 的决策比 Hermes 技能生命周期更长 |
| archive_days | 180 | 实验中 90 天 archive 在持续增长场景下过于激进 |
| pinned | 2% | 关键决策由用户手动标记，跳过自动流转 |

---

## Step 2: 上下文预取

### 业务价值
- **Token 效率 8 倍于全量注入**：76 tokens vs 731 tokens，效果达 85%
- **Embedding 排序是核心**：random 基准线仅 0.1 引用，prefetch 达 1.1 引用（11 倍）
- **每次查询都正收益**：embedding 计算成本固定，token 节省即时

### 数据证明
| 策略 | 引用数 | tokens | 效率 |
|------|--------|--------|------|
| random | 0.1 | 71 | 0.0014 |
| prefetch | 1.1 | 76 | **0.0144** |
| full | 1.3 | 731 | 0.0018 |

### 实现模式

```typescript
async function prefetchContext(
  query: string,
  knowledgeBase: KnowledgeStore,
  topK: number = 5
): Promise<ContextEntry[]> {
  // 1. 只在 active 记录中检索（与生命周期联动）
  const candidates = await knowledgeBase.searchActive(query, { limit: topK })
  // 2. 注入到 system prompt
  return candidates
}
```

### 观测指标
- `prefetch_hit_rate`：top-K 中有多少确实相关
- `avg_injected_tokens`：目标 < 500
- `refs_per_response`：LLM 响应中引用相关记录的次数

---

## Step 3: 上下文压缩（长会话场景）

### 业务价值
- **压缩质量远优于截断**：8.9% vs 44.9% 压缩比
- **7 次后续推理后回本**：前期净亏 33K tokens，长期节省

### 数据证明
| | 截断 | 3-phase |
|---|---|---|
| 压缩比 | 44.9% | **8.9%** |
| Token 成本 | 0 | 39,471 |
| 回本点 | — | ≥ 7 次推理 |

### 实现模式（3-Phase）

```
Phase 1 (Prune):    工具输出 > 200 字符 → 占位符
Phase 2 (Protect):  system prompt + 最近消息（60% budget）
Phase 3 (Summarize): LLM 结构化摘要（Active Task / Goal / Decisions / Remaining Work）
```

### 触发条件
- 对话 token 数 > context_window × 50%
- Session 步数 > 5（确保回本）

### 观测指标
- `compression_ratio`：目标 < 15%
- `tokens_saved_per_session`：累计节省
- `session_steps_post_compress`：压缩后剩余步数（需 > 7 才回本）

---

## Step 4: 知识合并（待验证）

### 当前状态
实验中 LLM 合并精确率 60%，低于纯 embedding 的 94.4%。但实验设计有三个根本性问题：

1. **候选集不对等**：embedding@0.70 只审了 18 对（最容易的），LLM 审了 50 对（含边界案例）。拿简单案例的精确率比困难案例的精确率，不公平
2. **高阈值 ≠ 好判断**：embedding 的"高精确率"本质上是保守策略（只合并非常相似的），不是判断能力更强
3. **Ground truth 太粗糙**：同集群 ≠ 应该合并。"缓存击穿用互斥锁"和"Pipeline 优化网络延迟"同属 Redis 集群但不该合并

另外 Hermes Curator 不做 embedding 预筛选——它直接让 LLM 审查全量候选。

### 重新评估条件
- 按 Hermes 原生方式重跑：无 embedding 预筛选，LLM 直接审查全量
- 用更精细的 ground truth（人工标注，而非"同集群=应合并"）
- 用下游检索质量作为最终评判，而非合并精确率

---

## 架构关系图

```
Agent Turn 开始
    │
    ├── [生命周期 tick] 每日一次，纯算法
    │       └── active ↔ stale → archived
    │       └── reactivation: stale → active（被引用时）
    │
    ├── [预取] 每次 turn
    │       └── embedding query → top-K active 记录
    │       └── 注入 system prompt
    │
    ├── [执行] Agent 推理 + 工具调用
    │       └── 引用的记录触发 touch()（刷新 lastReferencedAt）
    │
    └── [压缩] 对话超过阈值时
            └── prune → protect → summarize
            └── 压缩后继续推理
```

## 成本收敛证明

5 人团队预测（20 条/天，50 查询/天）：

| 月份 | Hermes 累计 ops | 无管理累计 ops | 节省 |
|------|----------------|--------------|------|
| 3 | 5.76B | 13.8B | 2.4x |
| 6 | 15.4B | 55.3B | 3.6x |
| 12 | 41.0B | 221B | **5.4x** |

第 12 个月无管理累计成本是 Hermes 的 5.4 倍，且差距持续扩大。
