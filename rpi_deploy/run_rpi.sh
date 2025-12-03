#!/bin/bash
# Run inference on Raspberry Pi (Two-Stage)

# Check requirements
# pip3 install -r requirements.txt

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Используйте пресеты моделей:
#   --model full    : обычная модель (быстрее, ~5.6MB)
#   --model quant   : квантованная модель (медленнее, ~1.6MB)
#   --model <path>  : путь к своей модели
#
# --threads 4: оптимально для Raspberry Pi 4 (4 ядра)
# Для других устройств: --threads 2 или --threads 1
python3 run_rpi.py \
    --input input \
    --output output \
    --model full \
    --threads 4


