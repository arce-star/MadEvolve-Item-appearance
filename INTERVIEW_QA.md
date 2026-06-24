# 面试问答：基于简历的深度追问

> 每个问题分三部分：面试官会怎么问 → 你怎么答 → 追问怎么应对

---

## Q1: "你说从零实现了回测引擎，具体做了哪些？"

**答**: 四个核心模块：
1. **Fill Model** — 被动限价单填充。每笔买单当 `Low < limit_price` 成交，卖单当 `High > limit_price` 成交。成交价 = limit price，无额外滑点。hit_ratio=1。
2. **PnL 分解** — `PnL_adj = 持仓盈亏 - 价差成本 - 15bps手续费 - 冲击成本`。每分钟计算一次，全年累加作为适应度。
3. **冲击模型** — 6 参数 Propagator Model。永久项 (α_perm) + 瞬时项 (α_trans × 幂律衰减核 G(τ))。自冲击保守计入（G(0)=1）。
4. **Alpha Forecaster** — 固定 Ridge(α=0.5)：3 个 EMA 特征 → 4 个 horizon 预测 → sum → 单一 α 信号。

**追问 "为什么选 impact-adjusted PnL 而不是 Sharpe 做适应度？"**
论文指出 Sharpe 可被"少交易"策略虚高——进化会偏向不交易。PnL_adj 内置冲击惩罚（~k^1.5），既惩罚过度交易又保持赚钱动力。

---

## Q2: "内存从 200MB 降到 4MB，具体怎么做的？"

**答**: 根因是 Python 对象开销太大：
- `pnl_components`: 40 万个 dict，每个 ~300B → 120MB。改为预分配 `np.zeros(527040, dtype=float64)` → 4MB。
- `trades`: 40 万个 TradeRecord dataclass 实例 ~80MB。改为 `int` 计数器（回测不需要存每笔交易详情）。

关键是意识到：回测打分只需要统计量（总 PnL、Sharpe、最大回撤），不需要保存每条明细。画图用的 equity curve 保留为简单 float list。

**追问 "还有哪些可以进一步优化？"**
主循环 52.7 万次 Python 迭代是当前瓶颈。用 numba JIT 编译或 Cython 可再提速 5-10 倍，但改动量较大。

---

## Q3: "LLM 缩进丢失的问题是什么？怎么解决的？"

**答**: DeepSeek 生成的 Python 代码缩进经常不正确。MadEvolve 框架把 LLM 输出直接插入到 EVOLVE-BLOCK 标记之间，如果标记在 class 体内（4 空格缩进），LLM 输出的代码却从 0 空格起头，`def` 就变成了顶层函数而非类方法，`self.set_limit_order()` 抛 AttributeError。

**解决**: 在 `blocks.py` 的 `replace_mutable_content()` 中增加：
1. 从原始 mutable 区域检测期望缩进（取所有非空行的最小缩进）
2. 从 LLM 输出检测其自身缩进
3. 重新缩进：LLM 缩进 - LLM 最小缩进 + 期望缩进

**追问 "这个问题好调试吗？"**
不好调试。score=0 没有报错——evaluator 的 try/catch 吞了 AttributeError 返回 None，策略默默不交易。最后通过对比 best.py 代码和 baseline 代码的缩进才定位到。

---

## Q4: "你的数据和论文不一样，结果还能比吗？"

**答**: 论文用 Polygon（付费聚合数据，BTCUSD），我用 Binance 公开数据（免费，BTCUSDT）。论文自己也写了 "do not expect quantitative results will hold out-of-the-box on any real exchange"——这篇论文本质是方法论验证，不是精确的实盘策略。

关键可比的是：
- 进化框架是否工作（是，快速版 -$44K → +$5.3K）
- 模型改进率是否合理（flash 100%, pro 33% — 符合论文"小模型高频微调+大模型偶尔创新"的结论）
- 策略是否做出了有意义的结构创新（是，LLM 自己加了 momentum 项、动态 no-trade band）

**追问 "如果你有 Polygon 数据，基线 Sharpe 会跟论文一样吗？"**
不会完全一样——论文基线 4.81，涉及很多我没复现的因素（聚合多交易所 fill rate、BTCUSD 价格水平等）。但方向一致即可。

---

## Q5: "MAP-Elites 在你的项目里具体怎么用的？"

**答**: 三个行为维度：
1. **complexity**: `len(code)` — 惩罚代码膨胀
2. **diversity**: 代码 embedding 与参考集的平均余弦距离 — 保持策略多样性
3. **performance**: 适应度分数

每个程序被分配到 10×10×10 的三维格子里，每格只保留最高分程序。父代选择时除了高分精英，还从不同格子里采样"结构不同但还可以"的策略作为灵感。

注意：我的 embedding 不是 OpenAI API 生成的（服务器访问不了），而是用 SHA256 哈希做的确定性伪向量——相同代码产生相同向量，不同代码产生不同向量。模型质量肯定不如真实 embedding，但不影响核心进化逻辑。

---

## Q6: "这个进化算法和传统遗传算法有什么区别？"

**答**: 
- **变异算子不是随机的** — 是 LLM 看了代码和指标后给出的语义级修改（"把静态阈值改成动态的"），不是 bit flip
- **种群管理更复杂** — 经典 GA 是固定大小的代际替换，MadEvolve 是稳态+精英制：种群持续增长，MAP-Elites 维持多样性，岛屿间定期迁移
- **不需要定义基因编码** — 直接对 Python 源码操作，LLM 理解代码语义

---

## Q7: "Run1 和 Run2 的区别？为什么 Run2 收益更大但优化更难？"

**答**:
- Run1 优化 `set_target`（alpha → 目标仓位，~70 行），涉及信号处理、仓位计算、风控判断
- Run2 优化 `set_limit_order`（目标仓位 → 限价单，~12 行），核心就一句 `mid * exp(-sign * depth)`

论文 Run2 的 PnL 提升 27×（vs Run1 的 6.4×），因为执行层的改进直接影响每笔交易的成交质量。但 Run2 代码空间极小（12 行），LLM 很难找到既不破坏语法又有收益的变异——我用 30 个候选没找到有效改进，论文用了 990 个才找到。

**追问 "如果你现在重新设计 Run2，会做什么不同的事？"**
把 EVOLVE-BLOCK 扩大，允许 LLM 在 `set_limit_order` 上方声明新的 UPPER_CASE 参数和辅助函数。论文的附录 C.2 描述的最优策略实际上就是 LLM 在 15 行函数体内塞了大量复杂逻辑（fill probability 模型、toxicity scoring、adaptive depth），所以更大的可变区域可能加速收敛。

---

## Q8: "这个项目还有什么没完成？接下来想做什么？"

**答**:
1. 全量进化实验正在跑（全年数据，50 候选 × 2 Run）
2. 跑完后需要画论文级别的对比图表（Cumulative PnL、进化进度、Sizing Decomposition）
3. OOS 测试：最优策略在 2025 测试集上的表现

如果时间充裕想做的：
- 加 Gemini API 做真正的多模型 ensemble（论文核心结论）
- 实现 Run 3（联合进化 target+limit）
- 用 Optuna 做超参校准（论文 Section 5.7）
- 加 numba 加速回测到 30 秒以内

---

## Q9: "你从中学到了什么？"

**答**: 
- **系统设计**: 如何设计评估协议的边界（子进程 vs import、result.json 格式、超时处理）
- **性能**: Python 对象的真实内存代价（一个 dict 300B，一个 int 28B，差 10 倍）；numpy 向量化的威力
- **AI 工程**: LLM 输出的不可靠性（缩进、语法、截断）需要在管道中加多层防御
- **科研诚实**: 知道自己的结果和论文的差距在哪，不夸大，诚实说明数据源、模型、规模的差异
