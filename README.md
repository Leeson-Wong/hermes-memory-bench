# Hermes 核心机制工程化验证

验证 Hermes Agent 三个核心记忆机制的实际可行性和性能表现。

## 角色

- **科学研究员**：设计假设、定义指标、分析数据、得出结论
- **架构师**：端口代码、搭建基础设施、确保可复现性

## 实验

| # | 名称 | 验证目标 |
|---|------|---------|
| 1 | [生命周期状态机](experiments/exp1-lifecycle/) | active→stale→archived 转换的准确性和时效性 |
| 2 | [知识合并](experiments/exp2-consolidation/) | LLM 驱动的 umbrella-building 合并质量 |
| 3 | [上下文预取](experiments/exp3-prefetch/) | prefetch 对响应相关性的提升效果 |
| 4 | [上下文压缩](experiments/exp4-compression/) | 3-phase 压缩的信息保留率和压缩比 |

## 环境设置

```bash
conda activate py312
pip install -r requirements.txt
```

## 运行

```bash
python run_experiment.py --exp 1        # 运行单个实验
python run_experiment.py --exp all      # 运行全部
python run_experiment.py --exp 1 --report  # 运行并生成报告
```
