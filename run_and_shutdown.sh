#!/bin/bash
# 运行 Run1 + Run2，全部结束后自动关机
set -e

cd /root/autodl-tmp/量化复现
source venv/bin/activate
export DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY"
export OPENAI_API_KEY="$DEEPSEEK_API_KEY"

mkdir -p logs

echo "=========================================="
echo "  MadEvolve Full Run + Auto Shutdown"
echo "  Start: $(date)"
echo "=========================================="

# 启动 Run1 (后台)
python -m madevolve run \
  -c code/config_run1_full.yaml \
  -o ./results_run1_full \
  -v > logs/run1_full.log 2>&1 &
PID1=$!

# 启动 Run2 (后台)
python -m madevolve run \
  -c code/config_run2_full.yaml \
  -o ./results_run2_full \
  -v > logs/run2_full.log 2>&1 &
PID2=$!

echo "Run1 PID: $PID1"
echo "Run2 PID: $PID2"
echo ""

# 监控两个进程，每 30 秒检查一次
while true; do
    RUN1_ALIVE=false
    RUN2_ALIVE=false

    if kill -0 $PID1 2>/dev/null; then
        RUN1_ALIVE=true
    fi
    if kill -0 $PID2 2>/dev/null; then
        RUN2_ALIVE=true
    fi

    if ! $RUN1_ALIVE && ! $RUN2_ALIVE; then
        echo ""
        echo "=========================================="
        echo "  Both runs finished at $(date)"
        echo "  Shutting down in 60 seconds..."
        echo "  (Ctrl+C to cancel)"
        echo "=========================================="
        sleep 60
        /usr/bin/shutdown -h now
        exit 0
    fi

    STATUS=""
    $RUN1_ALIVE && STATUS="$STATUS [Run1:running]"
    $RUN2_ALIVE && STATUS="$STATUS [Run2:running]"
    echo "$(date '+%H:%M:%S')$STATUS"
    sleep 30
done
