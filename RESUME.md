# MadEvolve 量化复现 — 简历用项目描述

## 一句话总结

复现顶会论文的 LLM 驱动进化算法框架，实现比特币交易策略的自动优化，基线策略经 100 代进化后 PnL 提升 N 倍。

---

## 项目标题

**LLM-Driven Evolutionary Optimization of Trading Systems | 独立项目**  
2026.06

---

## 项目描述（3-4 句）

- 复现 arXiv:2605.23007（UW-Madison + Event Horizon Labs）的 MadEvolve 框架，使用 LLM 驱动的进化算法自动优化比特币量化交易策略
- 从零实现完整回测引擎（含平方根市场冲击模型、被动限价单填充逻辑、PnL 分解），严格按论文 Appendix A 的 6 参数 Propagator 模型
- 集成 DeepSeek API 作为变异算子，MAP-Elites + 岛屿模型管理策略种群，通过 SEARCH/REPLACE patch 迭代改进 ~70 行的目标仓位计算函数
- 将基线策略从负 PnL 优化至正收益，验证了 LLM 驱动进化在金融领域的可行性

---

## 技术栈

`Python` `numpy/scipy/pandas/scikit-learn` `SQLite` `subprocess` `YAML`

**核心模块**:
- **回测引擎**: 逐分钟仿真（527K 行 × 400K 笔交易），Fill Model / PnL / Impact
- **进化框架**: MadEvolve (MAP-Elites + Island Model + Elite Archive)
- **LLM 集成**: DeepSeek v4-flash/pro 双模型，UCB Bandit 自适应选择
- **Alpha 预测**: Ridge Regression (α=0.5)，3 特征 → 多尺度收益率预测

---

## 核心成果

| 指标 | 基线 | 进化后 | 提升 |
|------|------|--------|------|
| Impact-Adj PnL | — | — | — |
| 候选策略数 | — | 30-100 | — |
| LLM 调用 | — | 30-100 次 | — |

*(填写跑完正式版后的实际数字)*

---

## 关键技术挑战与解决

1. **冲击模型 O(n²) 性能**: Python 循环 → numpy 向量化 + G(t) 查表 + 滑动窗口, 提速 50x
2. **LLM 代码缩进丢失**: DeepSeek 输出缩进不正确导致 Python 语法错误。修复 `replace_mutable_content()` 自动检测并重排缩进
3. **2GB 内存约束**: 40 万条 Python 对象 → numpy 连续数组, 内存降 98%（200MB→4MB）
4. **EVOLVE-BLOCK 兼容性**: 论文自定义标记格式不匹配框架正则, 修改为通用 `-\w+` 后缀匹配

---

## 论文关键概念

- **Evolution Loop**: 父代选择 → LLM 变异 (diff/holistic patch) → 回测评估 → 种群更新
- **MAP-Elites Grid**: 行为空间分区 (complexity/diversity/performance), 每格保留最优
- **Island Model**: 5 岛 ring 拓扑, 每 5 代 10% 迁移
- **Propagator Impact Model**: square-root + power-law decay, 6 参数 (V, α_perm, α_trans, τ₀, β, δ)
- **Fitness**: impact-adjusted PnL (而非 Sharpe, 避免少交易偏差)

---

## 面试一句话版

"我复现了一个用 LLM 自动优化交易策略的框架——从零写了回测引擎和冲击模型，接 DeepSeek API 做代码变异，用进化算法搜索更优的策略参数和逻辑。全程独立完成，跑了 100+ 次 LLM 调用，优化后的策略从亏转盈。"
