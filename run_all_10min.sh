#!/bin/bash
# 10-min alpha 全量三Run — 放后台跑完睡觉
cd /root/autodl-tmp/量化复现
source venv/bin/activate
export DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY"
export OPENAI_API_KEY="$DEEPSEEK_API_KEY"

FIG_MAP="code/visualize_mapelites.py"
FIG_ANALYZE="code/analyze_results.py"
FIG_POP="code/visualize_population.py"

echo "=== Run4 Full (100候选, ~5min) ===" | tee logs/all_10min.log
rm -rf results/run4_10min_full
python -m madevolve run -c code/config_run4_10min_full.yaml -o ./results/run4_10min_full -v 2>&1 | tee -a logs/all_10min.log
DIR4=$(ls -dt results/run4_10min_full/*/ 2>/dev/null | head -1)
[ -n "$DIR4" ] && python $FIG_ANALYZE "$DIR4" --run-name run4_10min --quick 2>&1 | tee -a logs/all_10min.log
[ -n "$DIR4" ] && python $FIG_MAP "$DIR4" --run-name run4_10min 2>&1 | tee -a logs/all_10min.log

echo "=== Run1 Full (50候选, ~3h) ===" | tee -a logs/all_10min.log
rm -rf results/run1_10min_full
python -m madevolve run -c code/config_run1_10min_full.yaml -o ./results/run1_10min_full -v 2>&1 | tee -a logs/all_10min.log
DIR1=$(ls -dt results/run1_10min_full/*/ 2>/dev/null | head -1)
[ -n "$DIR1" ] && python $FIG_ANALYZE "$DIR1" --run-name run1_10min --tiny 2>&1 | tee -a logs/all_10min.log
[ -n "$DIR1" ] && python $FIG_MAP "$DIR1" --run-name run1_10min 2>&1 | tee -a logs/all_10min.log
[ -n "$DIR1" ] && python $FIG_POP "$DIR1" --run-name run1_10min 2>&1 | tee -a logs/all_10min.log

echo "=== Run2 Full (50候选, ~1.5h) ===" | tee -a logs/all_10min.log
rm -rf results/run2_10min_full
python -m madevolve run -c code/config_run2_10min_full.yaml -o ./results/run2_10min_full -v 2>&1 | tee -a logs/all_10min.log
DIR2=$(ls -dt results/run2_10min_full/*/ 2>/dev/null | head -1)
[ -n "$DIR2" ] && python $FIG_ANALYZE "$DIR2" --run-name run2_10min --quick 2>&1 | tee -a logs/all_10min.log
[ -n "$DIR2" ] && python $FIG_MAP "$DIR2" --run-name run2_10min 2>&1 | tee -a logs/all_10min.log

echo "=== ALL DONE: $(date) ===" | tee -a logs/all_10min.log
