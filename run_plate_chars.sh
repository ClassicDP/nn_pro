#!/bin/bash
# Скрипт для запуска detect_plate_chars.py с активированным виртуальным окружением
# Оптимизировано для Raspberry Pi 4 с NEON инструкциями

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Активируем виртуальное окружение
source venv/bin/activate

# Устанавливаем переменные окружения для максимальной производительности на ARM
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OMP_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4

# Для ONNX Runtime
export ORT_DISABLE_ALL_OPTIMIZATIONS=0
export ORT_ENABLE_BASIC_OPTIMIZATIONS=1
export ORT_ENABLE_EXTENDED_OPTIMIZATIONS=1
export ORT_ENABLE_LAYOUT_OPTIMIZATIONS=1

# Устанавливаем приоритет для процесса (может помочь на загруженной системе)
# nice -n -5 дает более высокий приоритет (опционально)
# nice -n -5 python detect_plate_chars.py "$@"

# Запускаем скрипт с переданными аргументами
python detect_plate_chars.py "$@"

