#!/usr/bin/env python3
"""测试 v1 最优策略在 2025 测试集上的表现"""
import sys; sys.path.insert(0, 'code')
import numpy as np, pandas as pd, importlib.util
from quant_simulator import BacktestSimulator, AlphaForecaster

spec = importlib.util.spec_from_file_location(
    'best_v1', 'results/run1_full_v1_buggy/20260624_175827/best/best.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
s = mod.DefaultPassiveExecutor()

fc = AlphaForecaster()
fc.load('data/forecaster.pkl')
test = pd.read_parquet('data/btcusdt_1m_test.parquet')

# 第一周
sim = BacktestSimulator(test.iloc[:10080], fc)
r = sim.run(s)
print(f'V1 best on test (1st week 2025): \${r["combined_score"]:.0f}, trades={r["public_metrics"]["num_trades"]}')

# 全年
sim2 = BacktestSimulator(test, fc)
r2 = sim2.run(s)
print(f'V1 best on test (full 2025): \${r2["combined_score"]:.0f}, trades={r2["public_metrics"]["num_trades"]}')
