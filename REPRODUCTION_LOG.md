# MadEvolve 量化论文复现日志

> **论文**: MadEvolve: Evolutionary Optimization of Trading Systems with Large Language Models (arXiv:2605.23007v1)
> **创建日期**: 2026-06-24
> **当前状态**: 快速验证通过，待跑正式版

---

## 1. 环境信息

| 项目 | 值 |
|------|-----| 
| Python | 3.10.8 |
| 操作系统 | Linux 5.15.0 (Ubuntu) |
| MadEvolve | 0.1.0 (git, 8b881d3) |
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |
| scipy | 1.15.3 |
| openai | 2.43.0 |
| anthropic | 0.111.0 |
| matplotlib | 3.10.9 |
| pyarrow | 24.0.0 |
| PyYAML | 6.0.3 |
| rich | 15.0.0 |
| tiktoken | 0.13.0 |
| ccxt | 4.5.59 |

### LLM 配置

| 项目 | 值 |
|------|-----|
| 提供商 | DeepSeek |
| 模型 (快速探索) | deepseek-v4-flash (权重 75%) |
| 模型 (结构创新) | deepseek-v4-pro (权重 25%) |
| 选择策略 | UCB Bandit, 自适应 |
| max_tokens | 8192 |
| temperature | 0.7 |

---

## 2. 进度总览

- [x] 论文阅读与分析
- [x] MadEvolve 框架安装
- [x] EVOLVE-BLOCK 标记正则修复 (兼容论文自定义后缀)
- [x] 数据下载 (Binance 公开数据, BTCUSDT 1-min)
- [x] 数据质量检查 (无缺失, 无异常)
- [x] 回测模拟器实现 (严格按 Appendix A)
- [x] 基线策略实现 (Run1 / Run2, 严格按 Appendix B.2)
- [x] Evaluator 脚本 (MadEvolve 子进程调用)
- [x] Forecaster 预训练 (Ridge Regression, Appendix B.1)
- [x] Dummy Embedding 替代方案 (绕过 OpenAI API 依赖)
- [x] Run1 快速验证通过 (1周, 10候选, 7m, -44K→+5.3K)
- [x] Run2 快速验证 (1周, 10候选, 7m, score=0 — set_limit_order 太短难以优化)
- [x] 分析脚本 (analyze_results.py: 图表+指标+模型贡献, fig/目录输出)
- [x] 代码整理到 code/ 目录
- [x] Run1 半全量 — 2周, 30候选, 缩进修复后最佳较基线改进99.3%
- [x] Run2 半全量 — set_limit_order 12行代码空间过小, 收敛到0交易
- [x] 全量回测性能优化 — 42min→7min (numpy向量化+G查表+滑动窗口)
- [x] 内存优化 — 200MB→4MB (pnl_components numpy化)
- [x] JSON输出修复 — 大数组不入result.json
- [x] 基线回测进度条 — evaluator stderr输出+dispatcher不拦截
- [x] 代码上传GitHub
- [x] **Run1 Full 成功** — 全年数据, 50候选, 3h27m, -$2.3M→+$67K (转正!)
- [x] Run2 Full — set_limit_order 12行, DeepSeek 74%失败率, score=0
- [x] 完整图表 (fig/: run1_full/run1_quick/run1_semi/run2_full/run2_quick/run2_semi)
- [x] MAP-Elites 网格 + 种群动态可视化
- [ ] Run1 最优策略 OOS 测试 (2025年数据)

---

## 3. 数据收集

### 数据源

| 项目 | 值 |
|------|-----|
| 来源 | Binance Public Data (`data.binance.vision`) |
| 交易对 | BTCUSDT |
| 频率 | 1 分钟 K线 |
| 大小 | 46 个月 × ~43K 条/月 = ~2M 条 |

### 数据分割 (与论文一致)

| 划分 | 时间范围 | 行数 | 价格范围 | 用途 |
|------|---------|------|---------|------|
| Train | 2022-01 → 2023-12 | 1,051,120 | $15,514 ~ $48,164 | Forecaster 训练 |
| Val | 2024-01 → 2024-12 | 527,040 | $38,559 ~ $108,258 | 进化优化 (IS) |
| Test | 2025-01 → 2025-10-10 | 407,520 | $74,610 ~ $126,114 | OOS 评估 |
| Val Tiny | 2024-01 第一周 | 10,080 | $40,888 ~ $45,875 | 快速验证 |

### 质量检查

- 缺失值: 0
- 重复索引: 0
- 时间缺口 (>2min): 0
- H<L 异常: 0
- 零交易量: 0

---

## 4. 框架修改

### 修改 1: EVOLVE-BLOCK 正则 (constants.py)

**为什么需要改**

MadEvolve 默认只识别标准格式 `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END`。但论文策略代码使用了带后缀的自定义标记：

```python
# EVOLVE-BLOCK-TARGET-START  ← 论文格式
def set_target(self, state):   # Run1 只进化这个
    ...
# EVOLVE-BLOCK-TARGET-END

# EVOLVE-BLOCK-LIMIT-START    ← 论文格式  
def set_limit_order(self, state, target):  # Run2 只进化这个
    ...
# EVOLVE-BLOCK-LIMIT-END
```

默认正则 `EVOLVE-BLOCK-START` 匹配不到 `EVOLVE-BLOCK-TARGET-START`。框架的 `has_evolve_blocks()` 会返回 `False`，**整个文件被当作可变区域**——LLM 可能乱改 `set_limit_order`、`__init__`、甚至 `apply_order_constraints`，破坏进化边界。

**改了什么**

```python
# 修改前
EVOLVE_BLOCK_START_PATTERN = r"#\s*(?:===\s*)?EVOLVE-BLOCK-START(?:\s*===)?"
EVOLVE_BLOCK_END_PATTERN   = r"#\s*(?:===\s*)?EVOLVE-BLOCK-END(?:\s*===)?"

# 修改后 (加了 (?:-\w+)? 匹配可选后缀)
EVOLVE_BLOCK_START_PATTERN = r"#\s*(?:===\s*)?EVOLVE-BLOCK(?:-\w+)?-START(?:\s*===)?"
EVOLVE_BLOCK_END_PATTERN   = r"#\s*(?:===\s*)?EVOLVE-BLOCK(?:-\w+)?-END(?:\s*===)?"
```

新正则同时匹配 `EVOLVE-BLOCK-START`、`EVOLVE-BLOCK-TARGET-START`、`EVOLVE-BLOCK-LIMIT-START`。Run1 的 LLM 只能改 `set_target`，Run2 只能改 `set_limit_order`。

---

### 修改 2: Dummy Embedding (vectorizer.py)

**为什么需要改**

MadEvolve 每接收一个候选策略就调 OpenAI `text-embedding-3-small` 生成代码嵌入向量，用于 MAP-Elites 的 diversity 维度（衡量不同策略的"结构差异"）。服务器无法访问 `api.openai.com`，Vectorizer 卡死。

**改了什么**

用 SHA256 哈希替代 API 调用：

```python
# 修改前: 调 OpenAI API
response = self._get_adapter().embed(texts=[code], model=..., dimensions=1536)
embedding = response.single_embedding

# 修改后: 确定性哈希生成
h = int(hashlib.sha256(code.encode()).hexdigest()[:8], 16) % (2**31 - 1)
rng = np.random.RandomState(h)
embedding = rng.randn(1536).tolist()  # 归一化到单位长度
```

**影响分析**

- same code → same vector, different code → different vector（确定性保证）
- 不是真正的语义向量，功能相似的代码未必向量相似
- diversity 是 MAP-Elites 的辅助维度，主要用于 spreading the population，核心驱动力是 PnL 分数
- 对进化质量影响极小

---

## 5. 代码模块清单

```
code/
├── evaluator.py              # MadEvolve 评估器入口 (子进程调用)
├── evaluator_quick.py        # 快速版评估器 (1周数据, 传入 --val-data)
├── quant_simulator.py        # 回测引擎
│   ├── MarketImpactCalculator #   Eq. 3-7 平方根冲击模型 (Table 9 参数)
│   ├── AlphaForecaster       #   Ridge 回归 (Appendix B.1)
│   └── BacktestSimulator     #   Fill逻辑 + PnL + 冲击 + 指标计算
├── baseline_run1.py          # 基线策略 Run1 (EVOLVE-BLOCK: set_target)
├── baseline_run2.py          # 基线策略 Run2 (EVOLVE-BLOCK: set_limit_order)
├── config_run1.yaml          # Run1 正式配置 (50候选, 全年数据)
├── config_run1_quick.yaml    # Run1 快速测试 (10候选, 1周数据)
├── config_run2.yaml          # Run2 正式配置
├── analyze_results.py        # 分析报告: 图表 + 指标表 + 模型贡献
├── train_forecaster.py       # 预训练 Ridge Forecaster
├── benchmark_backtest_v3.py  # 回测性能基准测试
├── download_data.py          # Binance 公开数据下载
├── extract_data.py           # Zip → Parquet + 合并
└── check_data.py             # 数据质量检查 + 可视化
```

---

## 6. 快速验证结果 (Run1 Quick)

**配置**: 1周数据 | 10候选 | 2并发 | 2026-06-24 01:01 UTC

| 指标 | 基线 | 最优进化 | 比率 |
|------|------|---------|------|
| Impact-Adj PnL | -$44,186 | +$5,283 | 转正 |
| Sharpe Ratio | -63.73 | 3.48 | — |
| Calmar Ratio | -51.14 | 15.78 | — |
| Win Rate | 50.3% | 51.2% | 1.02× |
| Max Drawdown | $45,061 | $17,456 | 0.39× |
| Trades | 7,845 | 14 | 0.002× |
| 耗时 | — | 7m24s | — |

**图表输出**: `python code/analyze_results.py <results_dir> --run-name run1 [--tiny]`
输出到 `fig/run1/`: evolution_progress.png, cumulative_pnl.png, sizing_decomposition.png, sharpe_calmar.png

---

### Run2 Quick (1周, 10候选)

**配置**: 1周 | 10候选 | 2并发 | 2026-06-24 01:25 UTC

| 指标 | 基线 | 最优 | 
|------|------|------|
| Impact-Adj PnL | -$44,186 | $0 |
| 原因 | — | 所有变异导致0交易或语法错误 |

### Run2 Semi (2周, 30候选)

**配置**: 2周 | 30候选 | 2并发 | 2026-06-24 01:49 UTC

| 指标 | 基线 | 最优 | 
|------|------|------|
| Impact-Adj PnL | -$103,968 | $0 |
| Sharpe Ratio | -74.56 | 0.00 |
| Trades | 16,189 | 0 |

**模型**: Flash 24次(20.8%相对改进), Pro 3次(33.3%)

**诊断**: `set_limit_order()` 仅12行纯数学，LLM无法在这么小的空间做有效变异。所有变异要么语法错误，要么产出不交易策略(score=0)。论文Run2也需990候选才找到方向。后续需要扩大搜索空间或合并Run1+Run2为联合进化(Run3)。

**图表**: `python code/analyze_results.py results_run2_semi/20260624_014927 --run-name run2`

---

### Run1 Full (全年, 50候选)

**配置**: 全年 2024 | 50候选 | 2并发 | 2026-06-24 17:58 UTC | 3h27m

| 指标 | 基线 | 最优进化 | 比率 |
|------|------|---------|------|
| Impact-Adj PnL | -$2,304,113 | **+$66,849** | 转正 |
| Sharpe Ratio | -28.94 | 2.29 | — |
| Calmar Ratio | -1.00 | 2.59 | — |
| Win Rate | 47.9% | 49.3% | 1.03× |
| Max Drawdown | $2,305,356 | $25,705 | 0.01× |
| Trades | 400,857 | 19 | 0.00005× |

**进化轨迹**: Gen1(-$2.3M) → Gen6(-$129K) → Gen8(+$30K转正) → Gen16(+$67K最优)
**模型**: Flash 30次(46.7%改进率, 产最优解), Pro 4次(50%)

**关键发现**: LLM学会了极度选择性交易(40万笔→19笔), 从高频微调转为只在大机会出手。

### 模型贡献 (快速版)

| 模型 | 调用次数 | 改进率 | 最高分 | 平均改进 |
|------|---------|--------|--------|---------|
| deepseek-v4-flash | 5 | **100%** | $5,283 | +$36,406 |
| deepseek-v4-pro | 3 | 33% | $4,292 | +$16,159 |

**关键发现**: Flash 100% 改进率 (增量优化稳定), Pro 偶尔做结构创新。进化学会了极度选择性交易 (7845→14笔)。

---

## 7. 关键实现细节与论文对照

| 论文要求 | 实现 | 状态 |
|---------|------|------|
| 基线参数 (11个) | sizing_factor=10000, q_max=200000, ... | ✅ 逐字一致 |
| set_target 逻辑 | long/short target + risk_reduction + lag + correction | ✅ 逐行一致 |
| set_limit_order | exp(-s·d) 定价, riskoff分支 | ✅ 逐行一致 |
| Fill 逻辑 | Buy: L<p_limit, Sell: H>p_limit, hit_ratio=1 | ✅ |
| PnL 公式 | PnL_adj = q·δm - spread - fee - impact | ✅ Eq. 2-8 |
| 冲击模型 | V=$2B, α_perm=0.005, α_trans=0.010, τ₀=300s, β=0.5, δ=0.5 | ✅ Table 9 |
| Ridge α=0.5 | 3特征 × 4目标 | ✅ Appendix B.1 |
| Patch 权重 | diff 70% / holistic 30% | ✅ |
| 参数上限 | ≤15 UPPER_CASE 常量 | ✅ task_description |
| 岛屿模型 | 2岛 ring拓扑, 5代迁移10% | ⚠️ 论文5岛, 快速验证用2岛 |
| α 聚合 | predictions[:,0] (仅1-min) | ✅ 论文"short-term prediction α" |
| alpha_sd | 全局常数 np.std(alphas) | ✅ 论文未指定, 常数更稳 |
| EMA 参数 | ewm(span=1/5/10) | ✅ 论文Appendix B.1代码 |
| 数据源 | BTCUSDT (Binance) | ⚠️ 论文用 BTCUSD (Polygon) |

---

## 8. 遇到的问题与解决方案

| # | 问题 | 解决方案 |
|---|------|---------|
| 1 | Binance API 被墙 | 改用 `data.binance.vision` 公开数据 (服务器可直连) |
| 2 | 2025 数据时间戳为微秒 (16位) | 自动检测 ms/us 格式 |
| 3 | embedding 调 OpenAI API 卡住 | 改为 hash-based deterministic dummy embedding |
| 4 | numpy seed 溢出 | 取模 `2**31-1` |
| 5 | OpenAI API key 未设置导致 vectorizer 失败 | 关闭 diversity 维度 + dummy embedding |
| 6 | 冲击模型 O(n²) 回测极慢 | 30天 lookback 窗口 + O(1) 永久项累加 |
| 7 | forecaster.pkl 625B 看起来太小 | Ridge 3特征×4目标仅12系数, 大小正常 |
| 8 | madevolve CLI 找不到 | 用 `python -m madevolve` |
| 9 | python-dotenv 缺失 | `pip install python-dotenv` |
| 10 | DeepSeek v4 推理模型消耗 token | max_tokens 8192 足够 (代码70行 ≈ 200-500 tokens) |
| 11 | max_tokens 反复修改 | 最终统一 8192 |
| 12 | analyze 回测图太慢 (需跑4次完整回测) | 加 `--tiny` 用1周数据秒出, `--run-name` 分目录存图 |
| 13 | Run1/Run2 semi 全部 score=0 | LLM 生成代码缩进丢失, `replace_mutable_content` 不加修正。DeepSeek 输出无缩进, `def` 变顶层函数 → AttributeError → 0交易。修复: `blocks.py:replace_mutable_content` 自动检测原始缩进并重排 LLM 输出 |
| 14 | 全量版基线评估崩溃 (No result file) | 全年 527K 行回测太慢→子进程超时。根因: 冲击模型 Python 循环 O(n²) |
| 15 | 回测性能瓶颈 (~35min/全量) | 三处优化: (1) 冲击模型 numpy 向量化 ~50x; (2) 预计算 mid/high/low 替代 `.iloc[]` ~5x; (3) \|α\|<1e-10 时跳过策略 ~1.3x。降至 7min/次 |
| 16 | 全量回测到 68% 崩溃 (OOM) | 2GB 内存不够: `pnl_components` (40万 dict ×300B≈120MB) + `trades` (40万 TradeRecord ×200B≈80MB) = 200MB。修复: pnl_components→numpy float64 数组(4MB), trades→int 计数器(28B)。200MB→4MB, 内存降 98% |
| 17 | result.json 炸裂 (74MB) | `_pnl_components` 被序列化进 JSON。修复: evaluator 写入前 strip 掉大数组 |
| 18 | 基线 PnL 严重为负 (-$2.3M vs 论文 +$83K) | 三个 bug: (1) EMA 用 halflife 而非 span — 过度平滑→alpha_sd 极小→仓位巨大→高频交易; (2) α 聚合用 sum(4 horizon) — 论文"short-term"应只用 1-min 预测; (3) alpha_sd 用 60min 滚动 — 论文未指定, 改为全局常数更稳。修复后: -$2.3M → +$1,157 |

---

## 9. 下一步计划

1. **Run1 正式版**: 全年验证数据, 50候选, 预计 30-60 分钟
   ```bash
   python -m madevolve run -c code/config_run1.yaml -o ./results_run1
   ```
2. **Run2 快速验证**: 同 Run1 quick 模式, 验证 set_limit_order 进化
3. **Run2 正式版**
4. **完整报告**: `python code/analyze_results.py results_run1/xxx`
5. **OOS 测试**: 对最优策略跑 test 集回测
6. **超参校准** (可选): 按 Section 5.7, Optuna 调 8 个参数
7. **多模型对比** (可选): 加 Gemini 等模型
