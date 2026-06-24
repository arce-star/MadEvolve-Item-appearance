#!/usr/bin/env python3
"""解压 data/ 下的 Binance zip → 转 parquet → 合并 train/val/test"""
import os, zipfile, sys
from pathlib import Path
import pandas as pd

DATA_DIR = "/root/autodl-tmp/量化复现/data"
MONTHLY_DIR = os.path.join(DATA_DIR, "monthly")
os.makedirs(MONTHLY_DIR, exist_ok=True)

COLS = ["timestamp","open","high","low","close","volume",
        "close_time","quote_vol","trades","taker_buy_base","taker_buy_quote","ignore"]

SPLITS = {
    "train": ("2022-01-01", "2023-12-31"),
    "val":   ("2024-01-01", "2024-12-31"),
    "test":  ("2025-01-01", "2025-10-10"),
}

zips = sorted(Path(DATA_DIR).glob("BTCUSDT-1m-*.zip"))
print(f"找到 {len(zips)} 个 zip 文件")

ok = 0
for zp in zips:
    # 从文件名提取月份: BTCUSDT-1m-2022-01.zip → 202201
    month_str = zp.stem.split("-1m-")[1]  # "2022-01"
    parquet_path = os.path.join(MONTHLY_DIR, f"btcusdt_1m_{month_str.replace('-','')}.parquet")

    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        if len(df) > 100:
            print(f"  {month_str}: 跳过 (已有 {len(df)} 条)")
            ok += 1
            continue

    print(f"  {month_str}: 解压中 ...", end=" ", flush=True)
    try:
        import zipfile
        with zipfile.ZipFile(zp) as z:
            csv_name = z.namelist()[0]
            z.extract(csv_name, DATA_DIR)
        csv_path = os.path.join(DATA_DIR, csv_name)

        df = pd.read_csv(csv_path, names=COLS, header=None)
        df = df[["timestamp","open","high","low","close","volume"]]
        # 2022-2024: ms (13位), 2025+: us (16位)
        unit = "us" if df["timestamp"].iloc[0] > 1e15 else "ms"
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit=unit)
        df = df.set_index("timestamp").sort_index()
        df.to_parquet(parquet_path)
        os.remove(csv_path)  # 清理 CSV, 只留 parquet

        print(f"✓ {len(df)} 条")
        ok += 1
    except Exception as e:
        print(f"✗ {e}")

print(f"\n解压完成: {ok}/{len(zips)}")

# 合并
print("\n=== 合并数据 ===")
for sn, (start, end) in SPLITS.items():
    out_path = os.path.join(DATA_DIR, f"btcusdt_1m_{sn}.parquet")
    if Path(out_path).exists():
        df = pd.read_parquet(out_path)
        print(f"  {sn}: 已存在 {len(df):,} 条")
        continue

    dfs = []
    for fname in sorted(os.listdir(MONTHLY_DIR)):
        if not fname.endswith(".parquet"):
            continue
        df = pd.read_parquet(os.path.join(MONTHLY_DIR, fname))
        df = df[(df.index >= start) & (df.index <= end + " 23:59:59")]
        if len(df) > 0:
            dfs.append(df)

    if dfs:
        merged = pd.concat(dfs).sort_index()
        merged = merged[~merged.index.duplicated(keep="first")]
        merged.to_parquet(out_path)
        print(f"  {sn}: {len(merged):,} 条 → {Path(out_path).name}")
    else:
        print(f"  {sn}: 无数据!")

print("\n=== 统计 ===")
for sn in ["train", "val", "test"]:
    fp = os.path.join(DATA_DIR, f"btcusdt_1m_{sn}.parquet")
    if Path(fp).exists():
        df = pd.read_parquet(fp)
        print(f"  {sn}: {len(df):,} 条  {df.index[0]} → {df.index[-1]}  "
              f"price: {df['close'].min():.1f} ~ {df['close'].max():.1f}")
