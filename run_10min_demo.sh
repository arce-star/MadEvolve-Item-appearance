#!/bin/bash
# 10-min alpha — Demo 快速验证 (3个run依次跑)
set -e
cd /root/autodl-tmp/量化复现
source venv/bin/activate
export DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY"
export OPENAI_API_KEY="$DEEPSEEK_API_KEY"

echo "=== Run1 Demo (5候选, 2周数据) ==="
rm -rf results/run1_10min_demo
python -m madevolve run -c code/config_run1_10min_demo.yaml -o ./results/run1_10min_demo -v

echo "=== Run2 Demo (5候选, 2周数据) ==="
rm -rf results/run2_10min_demo
python -m madevolve run -c code/config_run2_10min_demo.yaml -o ./results/run2_10min_demo -v

echo "=== Run4 Demo (5候选, 1周数据) ==="
rm -rf results/run4_10min_demo
python -m madevolve run -c code/config_run4_10min_demo.yaml -o ./results/run4_10min_demo -v

echo "=== All demos done! ==="
echo "Results: results/run1_10min_demo/ results/run2_10min_demo/ results/run4_10min_demo/"
