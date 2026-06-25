#!/usr/bin/env python3
"""绘制基线累积 PnL 曲线 (α=[:,0], global α_sd)"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import importlib.util

from quant_simulator import BacktestSimulator, AlphaForecaster

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
VAL_PATH = os.path.join(DATA_DIR, 'btcusdt_1m_val.parquet')  # 全年数据, ~4min
FORECASTER_PATH = os.path.join(DATA_DIR, 'forecaster.pkl')

# 确保参数: span + [:,0] + global α_sd
# 如果 forecaster.pkl 不是这个版本, 删掉重训
need_retrain = True  # 强制重训

val = pd.read_parquet(VAL_PATH)
fc = AlphaForecaster()

if need_retrain or not os.path.exists(FORECASTER_PATH):
    print("训练 Forecaster (span, [:,0], global α_sd)...")
    train = pd.read_parquet(os.path.join(DATA_DIR, 'btcusdt_1m_train.parquet'))
    fc.fit(train)
    # Override predict to use global alpha_sd
    os.makedirs(DATA_DIR, exist_ok=True)
    fc.save(FORECASTER_PATH)
else:
    fc.load(FORECASTER_PATH)

# 加载基线策略
spec = importlib.util.spec_from_file_location('s', os.path.join(os.path.dirname(__file__), 'code', 'baseline_run1.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
s = mod.DefaultPassiveExecutor()

print(f"回测 {len(val):,} 行 ...")
sim = BacktestSimulator(val, fc, verbose=True)
r = sim.run(s)

pnl = r['combined_score']
sharpe = r['public_metrics']['sharpe_ratio']
print(f"PnL=${pnl:,.0f}  Sharpe={sharpe:.2f}")

# 画图
fig, ax = plt.subplots(figsize=(14, 5))
eq = np.array(r['_equity_curve'])
times = val.index[:len(eq)] if len(val) >= len(eq) else val.index
# 每小时采样一个点
step = max(1, len(eq) // 500)
ax.plot(times[::step], eq[::step], linewidth=1.0, color='#1B5E20')
ax.fill_between(times[::step], 0, eq[::step], alpha=0.1, color='#1B5E20')
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_title(f'Baseline Cumulative Impact-Adj PnL (α=1-min only, global α_sd)\nPnL=${pnl:,.0f}  Sharpe={sharpe:.2f}', fontsize=13)
ax.set_ylabel('Cumulative PnL (USD)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.grid(True, alpha=0.3)
plt.tight_layout()
ts = time.strftime('%Y%m%d_%H%M%S')
out = os.path.join(os.path.dirname(__file__), 'fig', f'baseline_cumulative_pnl_{ts}.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"图片: {out}")
