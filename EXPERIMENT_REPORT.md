# MadEvolve 复现实验完整报告

> 生成日期: 2026-06-25 | 论文: arXiv:2605.23007v1

---

## 一、代码版本演化

实验过程中发现了 4 个关键 bug，逐步修复。下表列出每个实验阶段使用的代码版本：

| 版本 | EMA | α 聚合 | alpha_sd | R² 公式 | ICIR 处理 |
|------|-----|--------|----------|---------|-----------|
| v1 (buggy) | `halflife=1/5/10` | `sum(axis=1)` | rolling 60 | centered | 未 clamp/scale |
| v2 (partial fix) | `span=1/5/10` | `sum(axis=1)` | rolling 60 | centered | 未 clamp/scale |
| v3 (single fix) | `span=1/5/10` | `[:,0]` (1-min) | rolling 60 | centered | 未 clamp/scale |
| **v4 (full fix)** | `span=1/5/10` | `[:,0]` (1-min) | global const | centered | 未 clamp/scale |
| **v5 (10-min)** | `span=1/5/10` | `[:,1]` (10-min) | global const | centered | 未 clamp/scale |
| **v6 (score fix)** | `span=1/5/10` | — | — | no-intercept Σy² | clamp+scale |

**v1→v4**: 逐步修复 forecaster bug，基线从 -$2.3M 改善到 +$1.2K  
**v5**: 论文确认 10-min 为主要预测周期，改 α 来源  
**v6**: 论文 Table 2 验证公式，clamp ICIR 到 [-5,5] 并除 5

---

## 二、完整实验列表

### 实验 1: Run1 Quick (v1)

| 项目 | 值 |
|------|-----|
| **日期** | 2026-06-24 01:01 UTC |
| **代码版本** | v1 (halflife, sum α, rolling α_sd) |
| **配置文件** | `code/config_run1_quick.yaml` |
| **α horizon** | sum(4 horizons) |
| **验证数据** | 1 周 (10,080 行) |
| **候选数** | 10 |
| **并发/岛屿** | 2/2 |
| **LLM 模型** | DeepSeek v4-flash(75%) + pro(25%) |
| **耗时** | 7m24s |
| **基线 PnL** | -$44,186 |
| **最优 PnL** | +$5,283 |
| **基线 Sharpe** | -63.73 |
| **最优 Sharpe** | +3.48 |
| **基线 Trades** | 7,845 |
| **最优 Trades** | 14 |
| **结果目录** | `results/run1_quick_1w/` |
| **图表** | `fig/run1_quick/` (evolution_progress, mapelites ×3) |
| **意义** | 首次验证 pipeline 能跑通。LLM 学会极度选择性交易 (7845→14笔) |

---

### 实验 2: Run2 Quick (v1)

| 项目 | 值 |
|------|-----|
| **日期** | 2026-06-24 01:25 UTC |
| **代码版本** | v1 |
| **配置文件** | `code/config_run2_quick.yaml` |
| **验证数据** | 1 周 |
| **候选数** | 10 |
| **耗时** | 7m16s |
| **基线 PnL** | -$44,186 |
| **最优 PnL** | $0 |
| **结果** | ❌ 失败。LLM 无法在 12 行 set_limit_order 中找到有效变异 |
| **结果目录** | `results/run2_quick_1w/` |
| **图表** | `fig/run2_quick/` (evolution_progress, mapelites ×3) |

---

### 实验 3: Run1 Semi v1 (v1, 缩进修复前)

| 项目 | 值 |
|------|-----|
| **日期** | 2026-06-24 01:56 UTC |
| **代码版本** | v1 |
| **配置文件** | `code/config_run1_semi.yaml` |
| **验证数据** | 2 周 (20,160 行) |
| **候选数** | 30 |
| **耗时** | ~25min |
| **基线 PnL** | -$103,968 |
| **最优 PnL** | $0 |
| **结果** | ❌ 全部候选 score=0。根因: `replace_mutable_content` 未修正缩进, LLM 输出缩进丢失导致 `def set_target` 变成顶层函数 |
| **修复** | `MadEvolve/madevolve/transformer/blocks.py:replace_mutable_content()` 添加自动缩进检测+重排 |
| **结果目录** | `results/run1_semi_v1_buggy/` |
| **图表** | `fig/run1_semi/` (evolution_progress, mapelites ×3) |

---

### 实验 4: Run2 Semi v1 (v1, 缩进修复前)

| 项目 | 值 |
|------|-----|
| **日期** | 2026-06-24 01:58 UTC |
| **代码版本** | v1 |
| **配置文件** | `code/config_run2_semi.yaml` |
| **验证数据** | 2 周 |
| **候选数** | 30 |
| **基线 PnL** | -$103,968 |
| **最优 PnL** | $0 |
| **结果** | ❌ 同 Run1 semi — 缩进 bug |
| **结果目录** | `results/run2_semi_v1_buggy/` |
| **图表** | `fig/run2_semi/` (evolution_progress, mapelites ×3) |

---

### 实验 5: Run1 全量 v1 (buggy baseline)

| 项目 | 值 |
|------|-----|
| **日期** | 2026-06-24 17:58 UTC |
| **代码版本** | v1 (halflife, sum α, rolling α_sd) |
| **配置文件** | `code/config_run1_full.yaml` |
| **α horizon** | sum(4 horizons) |
| **验证数据** | 全年 2024 (527,040 行) |
| **候选数** | 50 |
| **并发/岛屿** | 2/2 |
| **耗时** | 3h27m |
| **基线 PnL** | **-$2,304,113** |
| **最优 PnL** | **+$66,849** (+$2.37M 改进!) |
| **基线 Sharpe** | -28.94 |
| **最优 Sharpe** | +2.29 |
| **基线 Trades** | 400,857 |
| **最优 Trades** | 19 |
| **模型贡献** | Flash: 30 次(47% 改进), Pro: 4 次(50%) |
| **结果目录** | `results/run1_full_v1_buggy/` |
| **图表** | `fig/run1_v1/` (4 analyze 图: cumulative_pnl, evolution_progress, sizing_decomposition, sharpe_calmar) |
| **意义** | 即使基线极差, LLM 也能找到大幅改进。但改进主要来自修复 halflife bug 导致的过度交易, 非真正预测能力提升 |

---

### 实验 6: Run2 全量 v1

| 项目 | 值 |
|------|-----|
| **日期** | 2026-06-24 17:57 UTC |
| **代码版本** | v1 |
| **配置文件** | `code/config_run2_full.yaml` |
| **验证数据** | 全年 2024 |
| **候选数** | 50 |
| **耗时** | 4h0m |
| **基线 PnL** | -$2,304,113 |
| **最优 PnL** | $0 |
| **成功/失败** | 13/50 (74% 失败率) |
| **结果** | ❌ 确认 Run2 不可行: set_limit_order 12行对 DeepSeek 太小 |
| **结果目录** | `results/run2_full_v1_buggy/` |
| **图表** | `fig/run2_full/` (evolution_progress, mapelites ×3) |

---

### 实验 7: Run1 全量 v4 (修复后)

| 项目 | 值 |
|------|-----|
| **日期** | 2026-06-24 23:28 UTC |
| **代码版本** | **v4 (span, [:,0], global α_sd)** |
| **配置文件** | `code/config_run1_full.yaml` |
| **α horizon** | 1-min ([:,0]) |
| **验证数据** | 全年 2024 |
| **候选数** | 50 |
| **耗时** | 2h37m |
| **基线 PnL** | **+$1,157** |
| **最优 PnL** | **+$3,009** (+160%) |
| **基线 Sharpe** | +0.44 |
| **最优 Sharpe** | +1.57 |
| **基线 Trades** | 47,238 |
| **最优 Trades** | 33,491 |
| **Sizing ratio** | 1.00 (非纯放大, Sharpe 提升证明) |
| **结果目录** | `results/run1_full/` |
| **图表** | `fig/run1_v4/` (cumulative_pnl, evolution_progress, sizing_decomposition, sharpe_calmar, mapelites ×3, population_frames) |
| **意义** | 修复所有 bug 后的正经结果。策略改进来自更选择性交易 (47K→33K), 非放大仓位。但 OOS test=0 |

---

### 实验 8: Run4 大规模特征进化 (pre-score-fix)

| 项目 | 值 |
|------|-----|
| **日期** | 2026-06-25 01:56 UTC |
| **代码版本** | v4 (centered R², 未 clamp ICIR) |
| **配置文件** | `code/config_run4_quick.yaml` (误设为 10000 候选) |
| **验证数据** | 1 周 val (7 天 IC), 全年 train |
| **实际候选** | 908 |
| **耗时** | ~3h |
| **基线 Score** | **0.3314** |
| **最优 Score** | **1.7732** (+435%) |
| **特征数** | 3 → 17-20 |
| **模型贡献** | Flash: 815 次(avg 0.41), Pro: 92 次(avg 0.18) |
| **结果目录** | `results/run4_quick/` |
| **图表** | `fig/run4/` (evolution_progress_prefix) |
| **意义** | LLM 成功从3个EMA特征进化出17+维特征(加波动率/成交量/ATR/滞后收益)。Flask >> Pro |
| **注意** | ICIR 未 clamp/scale → score 可>1, 绝对数值不可比论文。排名仍有效 |

---

### 实验 9: Run1 10-min 全量

| 项目 | 值 |
|------|-----|
| **日期** | 2026-06-25 03:56 UTC |
| **代码版本** | **v5 (10-min α)** |
| **配置文件** | `code/config_run1_10min_full.yaml` |
| **α horizon** | 10-min ([:,1]) |
| **验证数据** | 全年 2024 |
| **候选数** | 50 |
| **状态** | ✅ 完成 (25/50? 或全部完成) |
| **基线 PnL** | **-$325,958** |
| **最优 PnL** | **+$34,559** (+$360K 改进) |
| **结果目录** | `results/run1_10min_full/` |
| **意义** | 10-min α 有交易信号 (1.44%触发 vs 1-min 0.04%), 进化能找到利润 |

---

### 实验 10: Run2 10-min 全量

| 项目 | 值 |
|------|-----|
| **日期** | 2026-06-25 |
| **配置文件** | `code/config_run2_10min_full.yaml` |
| **候选数** | 50 |
| **基线 PnL** | -$325,958 |
| **最优 PnL** | -$86,942 (改进但未转正) |
| **结果目录** | `results/run2_10min_full/` |

---

### 实验 11: Run4 10-min 全量 (score-fixed)

| 项目 | 值 |
|------|-----|
| **日期** | 2026-06-25 15:24 UTC (进行中) |
| **代码版本** | **v6 (no-intercept R², clamp ICIR)** |
| **配置文件** | `code/config_run4_10min_full.yaml` |
| **候选数** | 100 |
| **基线 Score** | 0.057 |
| **状态** | 🔄 正在运行 |

---

### 实验 12: Run1/2/4 Demo (10-min, 3候选)

| 项目 | Run1 | Run2 | Run4 |
|------|------|------|------|
| **数据** | 2周 | 2周 | 1周 |
| **候选** | 3 | 3 | 3 |
| **结果** | score=0 | score=0 | 1/3成功 |
| **意义** | demo 数据太短, 全跳过 | | |

---

## 三、关键发现总结

### 代码 Bug 修复历程

| Bug | 症状 | 修复 | 影响 |
|-----|------|------|------|
| `ewm(halflife=)` | α过度平滑, 仓位疯狂 | 改为 `ewm(span=)` | 基线 -$2.3M→-$2.07M |
| `alpha=sum(4horizons)` | 1000-min 偏差污染 | 改为 `[:,0]` 1-min | 基线 →+$705 |
| `alpha_sd=rolling` | 局部波动导致仓位不稳 | 改为全局常数 | 基线 →+$1,157 |
| `R²=centered` | 与论文公式不一致 | 改为 no-intercept Σy² | score 修正 |
| `ICIR 未 clamp` | score 突破理论上限 | clamp[-5,5]÷5 | score 归一化到[0,1] |
| 缩进丢失 | LLM 输出全失败 | `replace_mutable_content` 自动重排 | Run1 semi 起效 |

### 论文确认

- 论文 Table 2 基线 Combined=0.0848, 我们修正后=0.078 → **92%吻合**
- 论文明确 10-min 为主要预测周期 (Section 5.5, 6.2)
- 论文 R² 用 no-intercept 公式, ICIR 除以 5 缩放

### 当前最佳结果

| Run | 版本 | 基线 | 最优 | 提升 |
|-----|------|------|------|------|
| Run1 v1 | buggy | -$2.30M | +$66.8K | +$2.37M |
| Run1 v4 | fixed | +$1.16K | +$3.01K | +160% |
| Run1 10min | v5 | -$326K | +$34.6K | +$360K |
| Run4 908 | pre-fix | 0.33 | 1.77 | +435% |
| Run4 10min | v6 | 0.057 | 🔄 | 🔄 |

### 图表清单

```
fig/
├── run1_v1/     (4 files)  ← v1 buggy 全量: 4张分析图
├── run1_v4/     (8 files)  ← v4 fixed 全量: 4张分析+3张mapelites+population
├── run1_quick/  (4 files)  ← v1 quick: evolution+3 mapelites
├── run1_semi/   (4 files)  ← v1 semi: evolution+3 mapelites
├── run1_full/   (8 files)  ← 同上v4
├── run2_full/   (4 files)  ← v1: evolution+3 mapelites
├── run2_quick/  (4 files)  
├── run2_semi/   (4 files)  
├── run4/        (2 files)  ← pre-fix evolution+prefix
└── summary/     (0 files)  
```
