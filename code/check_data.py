#!/usr/bin/env python3
"""数据质量检查 + 可视化"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DATA_DIR = "/root/autodl-tmp/量化复现/data"

def load_data():
    dfs = {}
    for split in ["train", "val", "test"]:
        path = os.path.join(DATA_DIR, f"btcusdt_1m_{split}.parquet")
        if os.path.exists(path):
            dfs[split] = pd.read_parquet(path)
            print(f"  {split}: {len(dfs[split]):,} 条  {dfs[split].index[0]} → {dfs[split].index[-1]}")
        else:
            print(f"  {split}: 缺失!")
    return dfs

def check_quality(dfs):
    print("\n=== 质量检查 ===")
    for name, df in dfs.items():
        print(f"\n--- {name} ---")
        # 缺失值
        nulls = df.isnull().sum()
        if nulls.sum() > 0:
            print(f"  缺失值: {nulls[nulls > 0].to_dict()}")
        else:
            print("  缺失值: 无")

        # 异常价格 (OHLC 逻辑)
        bad_hl = (df["high"] < df["low"]).sum()
        bad_oc = ((df["open"] > df["high"]) | (df["open"] < df["low"])).sum()
        bad_cc = ((df["close"] > df["high"]) | (df["close"] < df["low"])).sum()
        print(f"  H<L异常: {bad_hl},  O越界: {bad_oc},  C越界: {bad_cc}")

        # 零值/负值
        zeros = ((df[["open","high","low","close"]] <= 0).sum())
        print(f"  零/负价格: open={zeros['open']}, high={zeros['high']}, low={zeros['low']}, close={zeros['close']}")

        # 时间间隔
        gaps = df.index.to_series().diff().dropna()
        expected = pd.Timedelta(minutes=1)
        large_gaps = gaps[gaps > expected * 2]
        if len(large_gaps) > 0:
            print(f"  大时间缺口 (>2min): {len(large_gaps)} 处")
            for t, v in large_gaps.head(5).items():
                print(f"    {t}: 间隔 {v}")
        else:
            print("  时间缺口: 无")

        # 重复
        dups = df.index.duplicated().sum()
        print(f"  重复索引: {dups}")

        # 交易量
        zero_vol = (df["volume"] <= 0).sum()
        print(f"  零交易量: {zero_vol} ({zero_vol/len(df)*100:.2f}%)")

        # 价格统计
        print(f"  价格范围: {df['close'].min():.1f} ~ {df['close'].max():.1f}")
        ret = df["close"].pct_change().dropna()
        print(f"  收益率: mean={ret.mean():.6%}, std={ret.std():.4%}, "
              f"min={ret.min():.4%}, max={ret.max():.4%}")
        print(f"  日波动率(年化): {ret.std() * np.sqrt(365*24*60):.1%}")

def plot_overview(dfs):
    print("\n=== 生成图表 ===")
    fig, axes = plt.subplots(3, 2, figsize=(20, 14))
    colors = {"train": "blue", "val": "green", "test": "red"}

    # 1. 完整价格走势
    ax = axes[0, 0]
    for name, df in dfs.items():
        ax.plot(df.index[::60], df["close"][::60], color=colors[name], alpha=0.7,
                linewidth=0.5, label=f"{name} ({len(df):,})")
    ax.set_title("BTCUSDT Close Price (hourly sampled)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 对数收益率分布
    ax = axes[0, 1]
    for name, df in dfs.items():
        ret = df["close"].pct_change().dropna() * 100
        ax.hist(np.clip(ret, -0.5, 0.5), bins=200, alpha=0.5, color=colors[name],
                label=f"{name} (σ={ret.std():.3f}%)", density=True)
    ax.set_title("1-Min Return Distribution (clipped ±0.5%)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Return (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 滚动波动率 (30天)
    ax = axes[1, 0]
    for name, df in dfs.items():
        if len(df) < 1440:
            continue
        vol = df["close"].pct_change().rolling(1440*30).std() * np.sqrt(365*24*60)
        ax.plot(df.index[::1440], vol[::1440], color=colors[name], alpha=0.8,
                linewidth=0.8, label=name)
    ax.set_title("30-Day Rolling Annualized Volatility", fontsize=13, fontweight="bold")
    ax.set_ylabel("Volatility")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 日均交易量
    ax = axes[1, 1]
    for name, df in dfs.items():
        daily_vol = df["volume"].resample("D").sum()
        ax.plot(daily_vol.index, daily_vol.values, color=colors[name], alpha=0.7,
                linewidth=0.5, label=name)
    ax.set_title("Daily Trading Volume (BTC)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Volume (BTC)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. 数据覆盖热力图 (按周)
    ax = axes[2, 0]
    all_df = pd.concat(dfs.values())
    all_df["week"] = all_df.index.isocalendar().week
    all_df["year"] = all_df.index.year
    all_df["day"] = all_df.index.dayofweek
    all_df["hour"] = all_df.index.hour
    coverage = all_df.groupby(["year", "week"]).size().unstack(level=0)
    # 期望每周 7*24*60 = 10080 分钟
    pct = coverage / 10080 * 100
    im = ax.imshow(pct.T, aspect="auto", cmap="RdYlGn", vmin=90, vmax=100)
    ax.set_title("Weekly Data Completeness (%)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Week of Year")
    ax.set_ylabel("Year")
    plt.colorbar(im, ax=ax)

    # 6. Split 时间线
    ax = axes[2, 1]
    ax.axis("off")
    lines = ["Data Split Summary", "=" * 30]
    for name, df in dfs.items():
        lines.append(f"\n{name.upper()}:")
        lines.append(f"  Period: {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")
        lines.append(f"  Candles: {len(df):,}")
        lines.append(f"  Price: ${df['close'].min():.0f} ~ ${df['close'].max():.0f}")
        total_vol = df["volume"].sum()
        lines.append(f"  Total Volume: {total_vol:,.0f} BTC")
        ret = df["close"].pct_change().dropna()
        lines.append(f"  Volatility: {ret.std()*np.sqrt(365*24*60):.1%} annualized")
    ax.text(0.1, 0.9, "\n".join(lines), transform=ax.transAxes,
            fontfamily="monospace", fontsize=10, verticalalignment="top")

    plt.tight_layout()
    out_path = os.path.join(DATA_DIR, "data_quality_report.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  图表: {out_path}")
    plt.close()

if __name__ == "__main__":
    dfs = load_data()
    if not dfs:
        print("无数据!")
        import sys; sys.exit(1)
    check_quality(dfs)
    plot_overview(dfs)
    print("\n完成!")
