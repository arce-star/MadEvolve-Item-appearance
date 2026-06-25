#!/bin/bash
# 补全所有缺失的图表
cd /root/autodl-tmp/量化复现
source venv/bin/activate

echo "=== Run1 v1 (buggy) — MAP-Elites + Population ==="
python code/visualize_mapelites.py results/run1_full_v1_buggy/20260624_175827 --run-name run1_v1
python code/visualize_population.py results/run1_full_v1_buggy/20260624_175827 --run-name run1_v1

echo "=== Run1 v4 (fixed) — 重新全部 (用正确label) ==="
python code/analyze_results.py results/run1_full/20260624_232809 --run-name run1_v4 2>/dev/null || echo "(running, skip)"

echo "=== Run1 10min — 全部 ==="
python code/analyze_results.py results/run1_10min_full/20260625_035639 --run-name run1_10min --tiny
python code/visualize_mapelites.py results/run1_10min_full/20260625_035639 --run-name run1_10min
python code/visualize_population.py results/run1_10min_full/20260625_035639 --run-name run1_10min

echo "=== Run2 10min — 全部 ==="
DIR2=$(ls -d results/run2_10min_full/*/ | head -1)
python code/analyze_results.py "$DIR2" --run-name run2_10min --quick
python code/visualize_mapelites.py "$DIR2" --run-name run2_10min

echo "=== Run4 908 — MAP-Elites + Population ==="
python code/visualize_mapelites.py results/run4_quick/20260625_015652 --run-name run4_908
python code/visualize_population.py results/run4_quick/20260625_015652 --run-name run4_908

echo "=== Run4 10min (running — will update when done) ==="

echo ""
echo "=== All done! ==="
echo "Check: fig/run1_v1/ fig/run1_10min/ fig/run2_10min/ fig/run4_908/"
