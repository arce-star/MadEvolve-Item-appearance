#!/usr/bin/env python3
"""Quick evaluator for fast pipeline validation (1 week val data)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force tiny validation data
_val = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "btcusdt_1m_val_2w.parquet")

# Inject --val-data into args
if "--val-data" not in sys.argv:
    sys.argv.insert(2, "--val-data")
    sys.argv.insert(3, _val)

# Run the real evaluator
import evaluator
evaluator.main()
