#!/bin/bash
# 整理项目目录结构
cd /root/autodl-tmp/量化复现

echo "=== 清理残留 ==="
rm -rf __pycache__ results_run1

echo "=== 整理 results/ ==="
mkdir -p results

# 移动并重命名旧结果
mv results_quick         results/run1_quick_1w   2>/dev/null || true
mv results_quick_run2    results/run2_quick_1w   2>/dev/null || true
mv results_run1_semi     results/run1_semi_2w    2>/dev/null || true
mv results_run2_semi     results/run2_semi_2w    2>/dev/null || true
mv results_run1_semi_v2  results/run1_semi_2w_v2 2>/dev/null || true
mv results_run2_semi_v2  results/run2_semi_2w_v2 2>/dev/null || true
mv results_run1_full     results/run1_full_12m   2>/dev/null || true
mv results_run2_full     results/run2_full_12m   2>/dev/null || true

echo "=== 最终结构 ==="
echo "量化复现/"
echo "├── code/           # 所有代码"
echo "├── data/           # 数据"
echo "├── MadEvolve/      # 框架"
echo "├── fig/            # 图表输出"
echo "├── results/        # 运行结果"
ls -d results/*/ 2>/dev/null | while read d; do echo "│   ├── $(basename $d)/"; done
echo "├── logs/"
echo "├── venv/"
echo "├── setup_env.sh"
echo "├── organize.sh"
echo "├── run_full.sh"
echo "├── run_and_shutdown.sh"
echo "├── *.md"
echo "└── quant_simulator.py  ← 保留主回测脚本(框架依赖)"
echo ""
echo "Done!"
