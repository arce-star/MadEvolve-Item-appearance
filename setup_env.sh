#!/bin/bash
# MadEvolve 量化复现 — 环境安装脚本
set -e

echo "=== Step 1: Create virtual environment ==="
cd /root/autodl-tmp/量化复现
python -m venv venv
source venv/bin/activate

echo "=== Step 2: Install MadEvolve (core) ==="
cd MadEvolve
pip install -e . 2>&1 | tail -5

echo "=== Step 3: Install optional deps (for full features) ==="
pip install -e ".[full]" 2>&1 | tail -5
pip install scipy pandas scikit-learn pyyaml ccxt pyarrow 2>&1 | tail -3

echo "=== Step 4: Verify imports ==="
python -c "
from madevolve import EvolutionOrchestrator, EvolutionConfig, PopulationConfig, ModelConfig, ExecutorConfig
print('MadEvolve import OK, version:', __import__('madevolve').__version__)
"

echo "=== Step 5: Set API keys ==="
export DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY"
echo "export DEEPSEEK_API_KEY=$DEEPSEEK_API_KEY" >> /root/autodl-tmp/量化复现/venv/bin/activate

echo ""
echo "=== Setup complete! ==="
echo "Run: source /root/autodl-tmp/量化复现/venv/bin/activate"
echo "DeepSeek API key is set in venv activate script."
