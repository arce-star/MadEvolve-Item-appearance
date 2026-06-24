#!/usr/bin/env python3
"""
Binance 公开数据下载 — BTCUSDT 1-min K线
数据源: https://data.binance.vision (无需 API Key, 服务器可直连)
"""

import os
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

SYMBOL = "BTCUSDT"
INTERVAL = "1m"
BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
COL_NAMES = [
    "timestamp", "open", "high", "low", "close", "volume",
    "close_time", "quote_vol", "trades", "taker_buy_base", "taker_buy_quote", "ignore"
]

SPLITS = {
    "train": ("2022-01-01", "2023-12-31"),
    "val":   ("2024-01-01", "2024-12-31"),
    "test":  ("2025-01-01", "2025-10-10"),
}


def download_month(month_str: str, cache_dir: str) -> str | None:
    """下载并解压一个月的 zip, 返回 CSV 路径"""
    zip_name = f"{SYMBOL}-{INTERVAL}-{month_str}.zip"
    zip_path = os.path.join(cache_dir, zip_name)
    csv_name = zip_name.replace(".zip", ".csv")
    csv_path = os.path.join(cache_dir, csv_name)

    # 已有 CSV, 跳过
    if os.path.exists(csv_path):
        return csv_path

    # 下载 zip
    url = f"{BASE_URL}/{SYMBOL}/{INTERVAL}/{zip_name}"
    if not os.path.exists(zip_path):
        print(f"  下载 {zip_name} ({month_str}) ...", end=" ", flush=True)
        try:
            resp = requests.get(url, stream=True, timeout=60)
            resp.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            size_mb = os.path.getsize(zip_path) / 1024 / 1024
            print(f"✓ {size_mb:.1f}MB")
        except Exception as e:
            print(f"✗ {e}")
            return None
    else:
        print(f"  已有 {zip_name}, 跳过下载")

    # 解压
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(cache_dir)
        return csv_path
    except zipfile.BadZipFile:
        print(f"  ✗ {zip_name} 损坏, 删除重试")
        os.remove(zip_path)
        return None


def process_month(month_str: str, cache_dir: str, monthly_dir: str) -> bool:
    """处理一个月: 下载 → 转 parquet → 保存到 monthly/"""
    parquet_path = os.path.join(monthly_dir, f"btcusdt_1m_{month_str.replace('-', '')}.parquet")
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        if len(df) > 100:
            print(f"  → 跳过 (已存在 {len(df)} 条)")
            return True

    csv_path = download_month(month_str, cache_dir)
    if csv_path is None:
        return False

    df = pd.read_csv(csv_path, names=COL_NAMES, header=None)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp").sort_index()

    os.makedirs(monthly_dir, exist_ok=True)
    df.to_parquet(parquet_path)
    print(f"  → {len(df)} 条 → {os.path.basename(parquet_path)}")
    return True


def merge_splits(data_dir: str):
    """合并月度 parquet 为 train/val/test"""
    monthly_dir = os.path.join(data_dir, "monthly")
    for split_name, (start, end) in SPLITS.items():
        out_path = os.path.join(data_dir, f"btcusdt_1m_{split_name}.parquet")
        if Path(out_path).exists():
            df = pd.read_parquet(out_path)
            print(f"  {split_name}: 已存在 {len(df):,} 条, 跳过")
            continue

        dfs = []
        for fname in sorted(os.listdir(monthly_dir)):
            if not fname.endswith(".parquet"):
                continue
            fp = os.path.join(monthly_dir, fname)
            df = pd.read_parquet(fp)
            df = df[(df.index >= start) & (df.index <= end + " 23:59:59")]
            if len(df) > 0:
                dfs.append(df)

        if dfs:
            merged = pd.concat(dfs).sort_index()
            merged = merged[~merged.index.duplicated(keep="first")]
            merged.to_parquet(out_path)
            print(f"  {split_name}: {len(merged):,} 条 → {os.path.basename(out_path)}")
        else:
            print(f"  {split_name}: 无数据!")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./data")
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()

    data_dir = args.output_dir
    cache_dir = os.path.join(data_dir, "cache")
    monthly_dir = os.path.join(data_dir, "monthly")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(monthly_dir, exist_ok=True)

    if not args.merge_only:
        months = pd.date_range("2022-01", "2025-10", freq="MS").strftime("%Y-%m").tolist()
        print(f"Binance 公开数据: {len(months)} 个月 ({months[0]} → {months[-1]})")
        print("=" * 50)

        ok = 0
        for m in months:
            print(f"\n[{ok + 1}/{len(months)}] {m}")
            if process_month(m, cache_dir, monthly_dir):
                ok += 1

        print(f"\n{'=' * 50}")
        print(f"完成: {ok}/{len(months)} 个月下载成功")

    print("\n=== 合并数据 ===")
    merge_splits(data_dir)

    print("\n=== 统计 ===")
    for sn in ["train", "val", "test"]:
        fp = os.path.join(data_dir, f"btcusdt_1m_{sn}.parquet")
        if Path(fp).exists():
            df = pd.read_parquet(fp)
            print(f"  {sn}: {len(df):,} 条  {df.index[0]} → {df.index[-1]}")


if __name__ == "__main__":
    main()
