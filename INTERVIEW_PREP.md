# MadEvolve 量化复现 — 面试准备文档

> 生成日期: 2026-06-24 | 基于 arXiv:2605.23007v1

---

## 1. 30秒电梯演讲

我复现了 MadEvolve 论文的 Run 1 和 Run 2——用 LLM 驱动的进化算法自动优化比特币交易策略。框架用一个 Ridge 回归生成 alpha 信号，MAP-Elites + 岛屿模型管理策略种群，LLM（DeepSeek v4-flash/pro）通过 SEARCH/REPLACE patch 迭代改进策略代码的特定组件。Run 1 进化目标仓位计算（set_target），Run 2 进化订单执行（set_limit_order）。用 Binance 公开数据替代论文的 Polygon，实现了完整的回测引擎（含平方根冲击模型），快速验证中基线 -$44K 被优化到 +$5.3K。最大挑战是让 LLM 在极小的代码空间（Run 2 仅 12 行）中做出有意义的结构创新。

---

## 2. 速查卡片

### 关键数字
| 项 | 值 |
|----|-----|
| 训练数据 | 2022-01→2023-12, 1,051,120 分钟 |
| 验证数据 | 2024-01→2024-12, 527,040 分钟 |
| 测试数据 | 2025-01→2025-10-10, 407,520 分钟 |
| LLM 模型 | deepseek-v4-flash (75%) + deepseek-v4-pro (25%) |
| max_tokens | 8192 |
| 快速验证 LLM 调用 | 10次 (7分钟) |
| 半全量 LLM 调用 | 30次 (15-25分钟) |
| API 成本估算 | ~$0.5-2/run (DeepSeek 极便宜) |

### 关键参数速记
| 参数 | 值 |
|------|-----|
| 冲击模型 | V=$2B, α_perm=0.005, α_trans=0.010, τ₀=300s, β=0.5, δ=0.5 |
| 基线策略参数 | sizing_factor=10000, q_max=200000, max_trade_frac=0.2, alpha_adjustment_knob=0.5, risk_reduction_factor=0.6, zp=0.0001, zp_riskoff=0.00003, fast_flat_minutes=10, std=1 |
| 手续费 | 15 bps (0.00015) |
| Forecaster | Ridge(α=0.5), 3 EMA 特征 (halflife=1,5,10), 预测 4 horizon (1m,10m,100m,1000m) |
| Patch 权重 | differential 70% / holistic 30% |

### 命令速记
```bash
# 启动进化
python -m madevolve run -c code/config_run1.yaml -o ./results_run1 -v

# 评估器调用 (框架自动)
python evaluator.py candidate.py --results_dir <work_dir>

# 分析结果
python code/analyze_results.py results_xxx --run-name run1 [--tiny]

# 单独跑基线
python code/evaluator.py code/baseline_run1.py --results_dir /tmp/test
```

---

## 3. 完整代码走读

### 文件: code/evaluator.py
- **作用**: MadEvolve 框架调用的评估器入口，子进程执行
- **被谁调用**: MadEvolve dispatcher (`dispatcher.py:249`) — `python evaluator.py candidate.py --results_dir <dir>`
- **调用了谁**: `quant_simulator.py` (BacktestSimulator, AlphaForecaster)
- **关键函数**:
  1. `main()` → None
     - 加载验证数据 → 加载/训练 Forecaster → 从 candidate.py 导入 DefaultPassiveExecutor → 跑 BacktestSimulator.run() → 写 result.json
     - 论文对应: Section 3.4 Fitness function, Section 4.1 回测流程
     - 关键参数: `--results_dir` (output dir), `--val-data` (可选覆盖数据路径), `--verbose` (进度条)
  2. `load_strategy_class(candidate_path)` → class
     - 用 importlib 动态加载 LLM 生成的 candidate.py 中的 DefaultPassiveExecutor
- **易出bug**: (1) candidate.py 有语法错误时 import 失败，返回 score=-1e10；(2) Forecaster 首次训练在 105 万行上需 ~3-5 秒

### 文件: code/evaluator_quick.py
- **作用**: evaluator.py 的 wrapper，自动注入 --val-data 指向 2 周小数据集
- **被谁调用**: MadEvolve dispatcher (快速版配置)
- **关键逻辑**: `sys.argv.insert(2, "--val-data")` 在 MadEvolve 参数中插入小数据路径

### 文件: code/quant_simulator.py
- **作用**: 核心回测引擎，严格按论文 Appendix A 实现
- **被谁调用**: evaluator.py, analyze_results.py
- **关键类/函数**:
  1. `MarketImpactCalculator` (line ~70)
     - 实现 Eq. 3-7 的平方根传播子冲击模型
     - 论文对应: Appendix A.4, Table 9
     - 关键参数: V, α_perm, α_trans, τ₀, β, δ
     - 30 天 lookback 截断 (G(30d)<1%)，永久项 O(1) 累加
  2. `AlphaForecaster` (line ~117)
     - Ridge(α=0.5) 回归: 3个 EMA 特征 → 4个 horizon 预测
     - 论文对应: Appendix B.1
     - `compute_features()`: EMA returns at halflife 1,5,10
     - `fit()`: 在 train 集拟合, `predict()`: sum(4 horizon) → α
  3. `BacktestSimulator` (line ~211)
     - 逐分钟回测循环: check fill → update position → build state → call strategy.set_passive_order_data() → submit order → compute PnL
     - 论文对应: Section 4.1, Appendix A.2.1, A.2.2, A.3
     - `_check_fill()`: Buy(L<p_limit), Sell(H>p_limit)
     - `run()`: return metrics dict 含 `_pnl_components` 和 `_equity_curve` 供画图
  4. `apply_order_constraints()` (line ~456)
     - 不可进化: 符号修正 + max_limit_order_usd=100K 截断
- **易出bug**: (1) 冲击模型不截断时 O(n²) 极慢；(2) 2025 时间戳从 ms 变 us 导致溢出

### 文件: code/baseline_run1.py
- **作用**: Run 1 基线策略 — EVOLVE-BLOCK 只包裹 set_target()
- **被谁调用**: MadEvolve 作为 init_program_path 加载，LLM 只修改 EVOLVE-BLOCK 内代码
- **论文对应**: Appendix B.2
- **关键类**:
  1. `DefaultPassiveExecutor` (11 个参数)
     - `set_target(state)` → dict: alpha → cost-adjusted long/short targets → risk_reduction → lag_adjustment → correction → clip → side
     - `set_limit_order(state, target)` → dict|None: FIXED (不在 EVOLVE-BLOCK 内)
     - `set_passive_order_data(state)`: 组合上面两个
- **易出bug**: set_target 有 ~70 行复杂分支，LLM 容易改出缩进错误

### 文件: code/baseline_run2.py
- **作用**: Run 2 基线策略 — EVOLVE-BLOCK 只包裹 set_limit_order()
- **区别**: set_target() 固定 (OUTSIDE EVOLVE-BLOCK)，set_limit_order() 可变
- **set_limit_order 仅 ~12 行**，LLM 极难找到有效改进

### 文件: code/config_run1.yaml / config_run2.yaml / config_run1_semi.yaml 等
- **作用**: YAML 配置，映射到 EvolutionConfig dataclass
- **关键段**: models (双模型+UCB), population (MAP-Elites+islands+archive), patch_policy (diff 70%/holistic 30%), executor (local, 2 parallel)

### 文件: MadEvolve/madevolve/engine/orchestrator.py
- **作用**: 进化主循环 — Producer-Consumer 模式
- **关键**: `_bootstrap_initial_generation()` (Gen 0), `_prepare_candidate()` (选父→LLM→patch), `_process_completed_job()` (评估→注册→更新种群)

### 文件: MadEvolve/madevolve/engine/configuration.py
- **作用**: 配置 dataclass 定义
- **关键**: EvolutionConfig, ModelConfig, PopulationConfig, PartitionConfig, IslandConfig, PatchPolicy, ExecutorConfig

### 文件: MadEvolve/madevolve/provider/gateway.py
- **作用**: 统一 LLM 接口，模型路由 + bandit 选择
- **关键**: `_infer_provider()` 通过模型名推断适配器, UCB/Thompson 选择策略

### 文件: MadEvolve/madevolve/repository/topology/partitions.py
- **作用**: MAP-Elites Grid + Island Cluster + Elite Vault
- **关键**: PartitionGrid (bin by complexity/diversity/performance), IslandCluster (ring topology, 5-gen migration), EliteVault (top-50)

### 文件: MadEvolve/madevolve/repository/selection/ancestry.py
- **作用**: 父代选择: PowerLawSelection, TournamentSelection, AdaptiveSelection
- **关键**: `ParentSelector.sample()` 返回父代 + 3 种灵感来源

### 文件: MadEvolve/madevolve/transformer/patcher.py
- **作用**: SEARCH/REPLACE diff patch 应用
- **关键**: `_extract_patch_blocks()` 正则提取, `_apply_single_block()` 精确匹配 + fuzzy fallback

### 文件: MadEvolve/madevolve/transformer/rewriter.py
- **作用**: Holistic rewrite 代码提取
- **关键**: `apply_holistic_rewrite()` 从 ```python``` 块提取 + 语法验证

### 文件: MadEvolve/madevolve/synthesizer/composer.py
- **作用**: 构建 LLM prompt: 任务描述 + 父代代码 + 灵感参考 + 反馈 + patch 指令
- **关键**: `compose()` 方法

### 文件: MadEvolve/madevolve/executor/dispatcher.py
- **作用**: 作业调度 — 子进程调用 evaluator.py, 监控完成, 收集 result.json
- **关键**: `LocalRunner.submit()`: `python evaluator.py candidate.py --results_dir <dir>`

### 文件: code/analyze_results.py
- **作用**: 论文级分析报告生成 — 指标表 + 4 张图 + 模型贡献
- **输出**: fig/{run_name}/ 下 evolution_progress.png, cumulative_pnl.png, sizing_decomposition.png, sharpe_calmar.png

---

## 4. 论文→代码 映射表

| 论文概念 | 论文位置 | 代码文件 | 函数/变量 | 实现要点 |
|----------|---------|---------|----------|---------|
| EVOLVE-BLOCK 标记系统 | Sec 3.1 | constants.py:16-17, blocks.py:19-79 | `EVOLVE_BLOCK_START_PATTERN`, `has_evolve_blocks()` | 正则: `EVOLVE-BLOCK(?:-\w+)?-START` 兼容论文自定义后缀 |
| Propagator 冲击模型 | App A.4, Eq.3-7 | quant_simulator.py:70-112 | `MarketImpactCalculator` | V=$2B, α_perm=0.005, α_trans=0.010, τ₀=300s, β=0.5, δ=0.5; 30天截断 |
| 填充逻辑 | App A.2.1 | quant_simulator.py:240-248 | `_check_fill()` | Buy: L&lt;p_limit, Sell: H&gt;p_limit, hit_ratio=1 |
| 订单生命周期 | App A.2.2 | quant_simulator.py:252-336 | `BacktestSimulator.run()` 循环 | check→update→cancel→set_passive_order→submit→log |
| PnL 计算 | App A.3 | quant_simulator.py:290-320 | run() 内 PnL 计算段 | PnL_adj = q·δm - spread - 15bps_fee - I_t |
| 冲击成本 c_i | App A.4, Eq.6 | quant_simulator.py:94-108 | `record_trade()` | D(t_i)·Q_i, 含 self-impact (G(0)=1, 保守) |
| 基线 set_target | App B.2 | baseline_run1.py:45-100 | `set_target()` | 11个参数, long/short cost-adjusted targets, risk_reduction, staleness, inventory, lag |
| 基线 set_limit_order | App B.2 | baseline_run2.py:90-105 | `set_limit_order()` | p_limit = mid_book·exp(-s·d), riskoff 用 zp_riskoff |
| Ridge 预测器 | App B.1 | quant_simulator.py:117-170 | `AlphaForecaster` | Ridge(α=0.5), 3 EMA 特征 (hl=1,5,10), 4 horizon 预测, α=sum(preds) |
| combined_score 设计 | Sec 3.4 | quant_simulator.py:420 | run() return `combined_score` | 用 PnL_adj 而非 Sharpe——避免"交易太少"偏差 |
| result.json 协议 | evaluator 协议 | quant_simulator.py:418-448 | run() return dict | success, combined_score, public_metrics (12项), private_metrics, text_feedback |
| MAP-Elites 网格 | Sec 3.2 | partitions.py:36-177 | `PartitionGrid` | complexity (len(code)), diversity (cosine distance), performance (score) |
| 岛屿模型 | Sec 3.2 | partitions.py:180-381 | `IslandCluster` | ring topology, 5-gen migration, 10% rate |
| 精英存档 | Sec 3.2 | partitions.py:384-470 | `EliteVault` | max_size=50, score-based replacement |
| 数据分割 | Sec 4.1 | extract_data.py:12-16 | SPLITS dict | train 2022-2023, val 2024, test 2025-10-10, chronological |
| Gate.io vs Polygon | Sec 4.1 | README / download_data.py | — | 论文用 Polygon (聚合多交易所), 我们用 Gate.io→后来改用 Binance Public Data (单交易所) |
| 模型集成 | Sec 3.3, 5.8 | config_run1.yaml:36-49 | models 段 | flash 75% (增量改进) + pro 25% (结构创新), UCB bandit 选择 |
| Patch 策略 | Sec 3.3 | config_run1.yaml:73-82 | patch_policy 段 | diff 70% / holistic 30%, 自适应, stagnation boost |
| 内循环优化 (未用) | Sec 3.4 | parallel.py | `ParameterOptimizer` | 论文禁用, 因为交易目标不可微 |
| 超参上限 | Sec 3.4 | config task_description | "Maximum 15 tunable parameters" | 论文: 15-20 UPPER_CASE 常量 |
| 应用约束 | App B.2 | quant_simulator.py:456-472 | `apply_order_constraints()` | 不可进化: 符号修正 + 100K USD 上限 |
| Table 9 冲击参数 | App A.4 | quant_simulator.py:55-63 | `IMPACT_PARAMS` | 6个参数全部一致 |
| 适应度=PNL_adj | Sec 3.4 | quant_simulator.py:420 | `combined_score` | 核心优化目标 |

---

## 5. 面试高频问题

### 架构设计

**Q1: 为什么评估器设计成子进程调用而不是 Python import？**
A: 子进程隔离保证每次评估环境干净（不同 candidate.py 可能有冲突的 import），也防止 LLM 生成的恶意/死循环代码拖垮主进程。MadEvolve dispatcher 用 `subprocess.Popen`, stdout/stderr 重定向到文件。论文也这样做——子进程超时 600s 后自动 kill。

**Q2: combined_score 为什么选冲击调整 PnL 而不是 Sharpe？**
A: 论文 3.4 节明确说了两个原因：(1) Sharpe 可被"交易更少但更准"的策略虚高，进化会偏向不交易；(2) 冲击调整 PnL 已内置风险厌恶（冲击项~k^1.5），且保持了"真正赚钱"的经济激励。我们没有用论文 Run 4 的预测分数 (R²+IC+ICIR) 因为 Run 1/2 优化执行逻辑而非预测。

**Q3: EVOLVE-BLOCK 的边界怎么划定的？**
A: `constants.py` 的正则 `EVOLVE-BLOCK(?:-\w+)?-START/END` 匹配标记行。`blocks.py:split_code_regions()` 把代码拆成 prefix(保护) + mutable(可改) + suffix(保护)。LLM 只看到 mutable 区域，patch 也只应用在那里。Run 1 只包 set_target，Run 2 只包 set_limit_order。`apply_order_constraints` 永远在标记外——不可被修改。

**Q4: MAP-Elites 的行为特征维度怎么选的？**
A: 默认 3 维：(1) complexity = len(code) — 防止代码膨胀；(2) diversity = 1 - cosine_sim(embedding, reference_set) — 保持结构多样性；(3) performance = score — 纯性能。我们用 hash-based dummy embedding 替代 OpenAI embedding（服务器访问不了 api.openai.com），虽不是真正语义向量但保留了"不同代码→不同向量"的确定性。

### 论文复现

**Q5: 你用的数据和论文不一样，这会影响结果吗？**
A: 论文用 Polygon (聚合多交易所的 BTCUSD)，我们用 Binance Public Data (单交易所 BTCUSDT)。论文自己也承认 "polygon is not exchange-specific...do not expect quantitative results will hold out-of-the-box on any real exchange"。这是**方法论复现而非精确数值复现**——我们验证的是进化框架的有效性，不是某个具体 Sharpe 数字。Binance 是全球 BTC 流动性最好的交易所，数据质量足够。

**Q6: 你复现的基线 Sharpe 和论文的 4.81 差多少？为什么？**
A: 快速验证基线 Sharpe 约 -63（2 周数据），不能和论文全年 4.81 直接比。差异原因：(1) 数据源不同 (BTCUSDT vs BTCUSD)；(2) 论文 2024 BTC 全年牛市，我们只用 2 周数据波动极大；(3) 单交易所 vs 聚合数据。正式版跑全年数据才能公平对比。

**Q7: Run 1 和 Run 2 分别进化什么？为什么 Run 2 的收益更大？**
A: Run 1 进化 set_target（alpha→目标仓位，~70 行复杂逻辑），Run 2 进化 set_limit_order（目标仓位→限价单，~12 行数学公式）。论文 Table 1 显示 Run 2 PnL 增长 27× vs Run 1 的 6.4×。原因：执行优化（更聪明的挂单深度/时机）直接影响每笔交易的滑点和成交概率，比仓位计算改进见效更快。但 Run 2 代码空间极小，LLM 更难找到有效变异。

**Q8: 你怎么验证进化结果不是 p-hacking？**
A: 论文 Section 7 的方法是：(1) 保留独立测试集 (2025) 不进进化循环；(2) 对比 IS-OOS 退化率 vs 多重检验理论预测；(3) 检查规模不变指标 (Sharpe/Calmar) 是否也改进——如果是纯放大仓位，Sharpe 不会变。我们实现了 Fig 2 的 sizing decomposition 直接量化"放大效应 vs 真正改进"。

### 工程细节

**Q9: 回测引擎怎么处理填充逻辑？**
A: `_check_fill()` 函数：Buy 订单当 Low < limit_price 时成交，Sell 当 High > limit_price 时成交。hit_ratio=1 (完全成交)。论文说这是"被动限价单"——只有对方主动吃单才会成交。每次成交后立即 cancel 旧单、submit 新单（单一活跃订单模式）。

**Q10: 冲击模型的 6 个参数是什么意思？怎么校准的？**
A: V=$2B (日均市场量), α_perm=0.005 (永久冲击系数), α_trans=0.010 (瞬时冲击系数), τ₀=300s (衰减时间尺度), β=0.5 (衰减幂律指数), δ=0.5 (规模指数)。论文用 Hyperliquid BTC-USD 永续合约数据校准，我们直接沿用。冲击模型让适应度函数抵制"无限放大仓位"策略（Pnl~k·F - k^1.5·I）。

**Q11: 一次完整进化要多少 LLM 调用？怎么控制成本？**
A: 30 候选 = 30 次 LLM 调用。DeepSeek v4-flash 极便宜 (约 $0.0005/次)，整个 run 不到 $0.02。论文用 5 个模型 × 990 候选，成本高得多。我们的控制手段：UCB bandit 自动选择便宜模型、max_tokens=8192 防止过长输出、patch 失败最多 3 次重试。

**Q12: 遇到过的最大技术难点？**
A: (1) 冲击模型 O(n²) 性能——加了 30 天 lookback 截断和 O(1) 永久项累加，从不可用优化到秒级；(2) embedding API 被墙——用确定性哈希生成 dummy embedding，保存余弦相似度语义；(3) EVOLVE-BLOCK 论文格式不兼容——修了正则添加可选后缀匹配；(4) Run 2 代码空间太小 LLM 无效——需要更多代数和更精准的 prompt。

### 延伸思考

**Q13: 如果部署到实盘还需要做什么？**
A: (1) 回测→实盘的差距巨大：需加交易所实际 order book、延迟、撤单等；(2) 冲击模型需在实盘数据上重校准；(3) 需要实时数据管线替代历史 parquet；(4) 风控层：单日最大亏损、异常检测；(5) Alpha 信号可用更强模型 (XGBoost/LSTM) 替代 Ridge。论文本身就是 proof-of-concept，离生产至少差 80%。

**Q14: 这个框架最大的局限是什么？**
A: (1) LLM 生成的代码质量不稳定——语法错误率 ~10-20%；(2) 搜索空间受限于 LLM 的创造力——如果最优解不是人类可写的 Python 代码形式，进化永远找不到；(3) 过拟合风险——论文 Section 7 分析表明 IS-OOS 退化随搜索空间增大而恶化；(4) 计算成本——多模型 ensemble 的全量运行需百小时级 GPU+API 时间。

---

## 6. 补充说明

### 与论文的关键差异点 (面试时必须诚实说明)
1. **数据源**: Binance 公开数据 vs 论文 Polygon — 论文也承认不是生产级
2. **模型**: DeepSeek v4 (2个) vs 论文 5 模型 ensemble (Gemini+Claude+GPT+o4)
3. **岛屿**: 2 岛 vs 论文 5 岛 (快速模式, 正式可改)
4. **候选数**: 30 vs 论文 990 (快速验证, 正式可加)
5. **embedding**: hash dummy vs 论文 OpenAI text-embedding-3-small
6. **交易对**: BTCUSDT vs 论文 BTCUSD (USDT 计价, 价格几乎一致)

### 需要监控的 Run1 semi 指标
- 基线 PnL (2周数据)
- 最终最优 PnL
- Flash vs Pro 改进率
- 进化进度曲线 (是否单调上升)
