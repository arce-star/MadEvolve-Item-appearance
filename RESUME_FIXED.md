# 简历修正版

\noindent \textbf{2. LLM驱动的量化交易系统进化优化 (独立复现顶会架构)} \hfill 2026.06
\begin{itemize}
    \item \textbf{框架复现与回测引擎}: 独立复现顶会论文 \textit{MadEvolve} (arXiv:2605.23007) 的进化优化框架, 从零实现逐分钟级 BTCUSDT 回测引擎（全年验证集 52.7 万行）, 严格按论文 Appendix A 实现被动限价单填充逻辑、PnL 分解公式与 6 参数平方根市场冲击模型（Propagator Model）。
    \item \textbf{性能优化}: 针对冲击模型 O(n²) 计算瓶颈, 通过 numpy 向量化替代 Python 循环、预计算 G(τ) 衰减查表、滑动窗口裁剪历史交易, 将单次全年回测从 42 分钟压缩至 7 分钟；将逐条 dict 存储转为连续数组, 内存占用降低约 98\%（~200MB→4MB）。
    \item \textbf{进化算法与 LLM 集成}: 接入 DeepSeek v4-flash/pro 双模型作为变异算子, 使用 MAP-Elites 行为网格 (complexity/diversity/performance) + 2 岛环形迁移模型 + 全局精英档案管理策略种群。LLM 通过 SEARCH/REPLACE diff patch 或完整重写对目标函数做定向改进。
    \item \textbf{LLM 容错}: 针对 DeepSeek 输出缩进丢失导致策略代码语法错误的问题, 在框架的代码重组模块中增加了自动检测父类缩进上下文并重排 LLM 输出的机制。
    \item \textbf{实验结果}: 快速验证阶段 (1 周数据, 10 候选) 将基线策略从 -$44K 优化至 +$5.3K (转正)。半全量实验 (2 周数据, 30 候选) 中最佳候选较基线改善 99.3\%。全量进化实验 (全年数据, 50 候选/run, 双 run 并行) 正在进行中。
    \item \textbf{数据}: 使用 Binance 公开历史数据 (data.binance.vision) 的 BTCUSDT 交易对替代论文的 Polygon BTCUSD, 时间切分与论文一致 (train 2022-2023 / val 2024 / test 2025)。
\end{itemize}

---

## 声明上的注意事项（面试时主动提及）

1. **数据源不同**: 论文用 Polygon (聚合多交易所 BTCUSD), 我用 Binance 公开数据 (单交易所 BTCUSDT)。论文自己也说 "do not expect quantitative results will hold out-of-the-box on any real exchange"，这是方法论复现而非精确数值复现。

2. **模型不同**: 论文用 5 个模型 (Gemini/Claude/GPT/o4), 我用 2 个 DeepSeek 模型。论文的核心结论是 "多模型 ensemble 优于单模型"，我用双模型部分验证了这一结论 (flash 100% 改进率 vs pro 33%)。

3. **岛屿数**: 我用 2 岛, 论文用 5 岛。这是计算资源约束下的调整，不影响方法验证。

4. **全量还在跑**: 快速和半全量的结果已证明 pipeline 可行, 全量结果尚未产出。面试时诚实说 "in progress"。

5. **Forecaster 固定**: Run 1/2 中 Ridge 回归特征固定（3 个 EMA），不做特征工程。论文 Run 4/5 才进化特征。
