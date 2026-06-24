#!/bin/bash
# MadEvolve 全量 Run1 + Run2
# 预计: Run1 ~3h, Run2 ~1.5h
set -e

cd /root/autodl-tmp/量化复现
source venv/bin/activate
export DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY"
export OPENAI_API_KEY="$DEEPSEEK_API_KEY"

echo "=========================================="
echo "  MadEvolve Full Run — Run1 + Run2"
echo "  Run1: 100 candidates, full 2024 val"
echo "  Run2: 50 candidates, full 2024 val"
echo "  Start: $(date)"
echo "=========================================="

echo ""
echo ">>> Starting Run1 (set_target) ..."
python -m madevolve run \
  -c code/config_run1_full.yaml \
  -o ./results_run1_full \
  -v 2>&1 | tee logs/run1_full.log

echo ""
echo ">>> Run1 done at $(date)"
echo ">>> Starting Run2 (set_limit_order) ..."

python -m madevolve run \
  -c code/config_run2_full.yaml \
  -o ./results_run2_full \
  -v 2>&1 | tee logs/run2_full.log

echo ""
echo "=========================================="
echo "  All done! $(date)"
echo "  Results: results_run1_full/ + results_run2_full/"
echo "=========================================="
