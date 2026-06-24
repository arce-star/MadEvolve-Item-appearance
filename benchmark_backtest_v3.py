#!/usr/bin/env python3
"""
回测性能基准测试 V3 — 向量化冲击模型 + 预计算 + 快速路径
用法: python benchmark_backtest_v2.py
输出: results/bench_v3/  (不冲突旧版)
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))
import pandas as pd
from quant_simulator import BacktestSimulator, AlphaForecaster
import importlib.util

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
VAL_PATH = os.path.join(DATA_DIR, 'btcusdt_1m_val.parquet')
FORECASTER_PATH = os.path.join(DATA_DIR, 'forecaster.pkl')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'bench_v3')
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("  Benchmark V3 — 向量化冲击 + 预计算 + 快速路径")
print("  " + "=" * 56)

val = pd.read_parquet(VAL_PATH)
fc = AlphaForecaster()
fc.load(FORECASTER_PATH)

spec = importlib.util.spec_from_file_location(
    'strategy',
    os.path.join(os.path.dirname(__file__), 'code', 'baseline_run1.py')
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

sizes = [12]  # 直接跑全年

for n in sizes:
    if n == 12:
        df = val
        label = '12m_full'
    else:
        end = f'2024-{n:02d}-28 23:59:00'
        df = val[val.index <= end]
        label = f'{n}m'

    print(f"\n[V3] {label}: {len(df):,} 行 (每10万行汇报一次)")
    print("-" * 40)
    s = mod.DefaultPassiveExecutor()
    sim = BacktestSimulator(df, fc, verbose=True)  # verbose: 每行刷新会慢很多
    t0 = time.time()
    r = sim.run(s)
    t = time.time() - t0

    # 保存结果
    result = {
        'version': 'v3_sliding_window',
        'rows': len(df),
        'time_sec': t,
        'trades': r['public_metrics']['num_trades'],
        'pnl': r['combined_score'],
        'sharpe': r['public_metrics']['sharpe_ratio'],
    }
    import json
    with open(os.path.join(OUT_DIR, f'bench_{label}.json'), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n[V3] {label} => {t:.1f}s, trades={r['public_metrics']['num_trades']}, PnL=${r['combined_score']:,.0f}")

print("\n" + "=" * 60)
print("  V3 Benchmark Done! Results: results/bench_v3/")
print("=" * 60)
