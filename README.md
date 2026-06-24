# MadEvolve 量化复现

复现论文 [*MadEvolve: Evolutionary Optimization of Trading Systems with Large Language Models*](https://arxiv.org/abs/2605.23007) (UW-Madison, arXiv:2605.23007v1)。

使用 **LLM 驱动的进化算法** 自动优化比特币量化交易策略。

---

## 快速概览

```
BTC历史数据 → Ridgeα预测器 → 基线策略 → LLM进化循环 → 分析画图
                    ↓                      ↓
              3个EMA特征         MAP-Elites + 岛屿模型
              固定不变           DeepSeek v4-flash/pro
```

**核心思路**：把交易策略的 Python 代码作为"基因"，LLM 作为"变异算子"，回测 PnL 作为"适应度"。通过迭代选择父代 → LLM 修改代码 → 回测打分 → 存入种群，自动发现更好的策略。

---

## 项目结构

```
├── code/                       # 所有核心代码
│   ├── quant_simulator.py      #   回测引擎 (Fill/PnL/冲击模型/指标)
│   ├── evaluator.py            #   MadEvolve 评估器入口 (子进程)
│   ├── evaluator_quick.py      #   快速版 (小数据集验证)
│   ├── baseline_run1.py        #   基线策略 Run1 (EVOLVE-BLOCK: set_target)
│   ├── baseline_run2.py        #   基线策略 Run2 (EVOLVE-BLOCK: set_limit_order)
│   ├── config_run1_full.yaml   #   Run1 全量配置
│   ├── config_run2_full.yaml   #   Run2 全量配置
│   ├── config_run1_quick.yaml  #   Run1 快速测试
│   ├── config_run2_quick.yaml  #   Run2 快速测试
│   ├── config_run1_semi.yaml   #   Run1 半全量
│   ├── config_run2_semi.yaml   #   Run2 半全量
│   ├── analyze_results.py      #   结果分析 + 画图
│   ├── train_forecaster.py     #   预训练 Ridge 预测器
│   ├── benchmark_backtest_v3.py  #  回测性能基准
│   ├── download_data.py        #   Binance 公开数据下载
│   ├── extract_data.py         #   Zip→Parquet+合并
│   └── check_data.py           #   数据质量检查
│
├── MadEvolve/                  # 进化框架 (含少量修改)
│   └── madevolve/
│       ├── engine/orchestrator.py   # 进化主循环
│       ├── engine/configuration.py  # 配置 dataclass
│       ├── repository/topology/     # MAP-Elites + Island
│       ├── repository/selection/    # 父代选择
│       ├── transformer/             # Patch 应用
│       ├── provider/                # LLM 适配器
│       └── common/constants.py      # EVOLVE-BLOCK 正则 (已修改)
│
├── fig/                        # 图表输出
│   ├── run1/                   #   Run1 专属图
│   ├── run2/                   #   Run2 专属图
│   └── summary/                #   汇总对比图
│
├── REPRODUCTION_LOG.md         # 完整复现日志
├── setup_env.sh                # 环境安装脚本
├── run_full.sh                 # 全量运行脚本
└── run_and_shutdown.sh         # 运行+自动关机
```

---

## 安装

```bash
# 1. 克隆
git clone git@github.com:arce-star/MadEvolve-Item-appearance.git
cd MadEvolve-Item-appearance
```

```bash
# 2. 安装环境
bash setup_env.sh
source venv/bin/activate
```

```bash
# 3. 设置 API Key
export DEEPSEEK_API_KEY="sk-xxx"
export OPENAI_API_KEY="sk-xxx"   # 仅用于 embedding (已改为本地hash, 可选)
```

```bash
# 4. 下载数据 (从 Binance 公开数据)
mkdir -p data
# 将 BTCUSDT 1-min monthly zip 放入 data/ 后运行:
python code/extract_data.py
```
 
---

## 运行

### 快速测试 (2周数据, 30候选, ~25分钟)

```bash
python -m madevolve run -c code/config_run1_semi.yaml -o ./results_run1_semi -v
```

### 全量运行 (全年数据, 50候选, ~3小时)

```bash
python -m madevolve run -c code/config_run1_full.yaml -o ./results_run1_full -v
python -m madevolve run -c code/config_run2_full.yaml -o ./results_run2_full -v
```

### 分析结果

```bash
python code/analyze_results.py results_run1_full/<timestamp> --run-name run1
```

### 训练 Forecaster (可选, evaluator 会自动训)

```bash
python code/train_forecaster.py
```

---

## 框架修改说明

相比原版 MadEvolve, 做了以下修改:

| 文件 | 修改 | 原因 |
|------|------|------|
| `common/constants.py` | EVOLVE-BLOCK 正则改为 `-\w+` 后缀匹配 | 兼容论文自定义标记 (TARGET/LIMIT) |
| `provider/vectorizer.py` | embed() 改为 hash-based dummy | 服务器无 OpenAI API 访问 |
| `transformer/blocks.py` | replace_mutable_content() 加缩进自动修正 | DeepSeek 输出缩进不正确 |

---

## 实验结果

### 快速验证 (1周数据, 10候选)

| 指标 | 基线 | 最优进化 |
|------|------|---------|
| Impact-Adj PnL | -$44,186 | +$5,283 |

### 半全量验证 (2周数据, 30候选, 缩进修复后)

Run1: 基线 -$103,968 → 最优 -$691 (改进 99.3%)
Run2: set_limit_order 代码空间过小 (12行), LLM 难以找到有效变异

### 全量实验 (全年数据, 50候选)

🔄 进行中...

---

## 论文概念映射

| 论文 | 代码 |
|------|------|
| Propagator 冲击模型 (App A.4, Table 9) | `quant_simulator.py:MarketImpactCalculator` |
| 填充逻辑 (App A.2.1) | `quant_simulator.py:_check_fill()` |
| PnL_adj 计算 (App A.3) | `quant_simulator.py:run()` |
| Ridge α=0.5 预测器 (App B.1) | `quant_simulator.py:AlphaForecaster` |
| 基线策略 (App B.2) | `baseline_run1.py` / `baseline_run2.py` |
| MAP-Elites + Island + Elite (Sec 3.2) | `partitions.py` |
| Evolution Loop (Sec 3.1) | `orchestrator.py` |
| LLM Ensemble (Sec 3.3) | `gateway.py` |
| Evaluator 协议 | `evaluator.py` |

---

## 技术栈

`Python 3.10` `numpy` `scipy` `pandas` `scikit-learn` `SQLite` `YAML` `matplotlib`

LLM: DeepSeek v4-flash + v4-pro (UCB Bandit 自适应选择)

---

## 许可证

MIT

---

## 参考

- 论文: [arXiv:2605.23007](https://arxiv.org/abs/2605.23007)
- 原始框架: [MadEvolve](https://github.com/tianyi-stack/MadEvolve)
- 官网: [madevolve.org](https://madevolve.org)
