#!/bin/bash
# 10-min alpha — 全量 Run1+Run2+Run4
cd /root/autodl-tmp/量化复现
source venv/bin/activate
export DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY"
export OPENAI_API_KEY="$DEEPSEEK_API_KEY"

echo "=== Run1 Full (50候选, 全年, ~3h) ==="
rm -rf results/run1_10min_full
python -m madevolve run -c code/config_run1_10min_full.yaml -o ./results/run1_10min_full -v 2>&1 | tee logs/run1_10min_full.log

echo "=== Run2 Full (50候选, 全年, ~1.5h) ==="
rm -rf results/run2_10min_full
python -m madevolve run -c code/config_run2_10min_full.yaml -o ./results/run2_10min_full -v 2>&1 | tee logs/run2_10min_full.log

echo "=== Run4 Full (100候选, 全年, ~5min) ==="
rm -rf results/run4_10min_full
python -m madevolve run -c code/config_run4_10min_full.yaml -o ./results/run4_10min_full -v 2>&1 | tee logs/run4_10min_full.log

echo "=== All full runs done! ==="
