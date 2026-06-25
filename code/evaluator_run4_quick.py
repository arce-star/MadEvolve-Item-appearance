#!/usr/bin/env python3
"""Run4 Quick Evaluator — 1周 val 数据, 秒级评估"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_val = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "btcusdt_1m_val_tiny.parquet")
if "--val-data" not in sys.argv:
    sys.argv.insert(2, "--val-data")
    sys.argv.insert(3, _val)
import evaluator_run4
evaluator_run4.main()
