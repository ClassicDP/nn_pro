#!/bin/bash
# Автоматическая конвертация ONNX -> NCNN и тестирование производительности

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Активация venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Поиск onnx2ncnn
ONNX2NCNN=""
POSSIBLE_PATHS=(
    "$HOME/ncnn/build/tools/onnx/onnx2ncnn"
    "/usr/local/bin/onnx2ncnn"
    "/usr/bin/onnx2ncnn"
    "onnx2ncnn"
)

for path in "${POSSIBLE_PATHS[@]}"; do
    if [ -f "$path" ] && [ -x "$path" ]; then
        ONNX2NCNN="$path"
        echo "✓ Найдена утилита: $ONNX2NCNN"
        break
    fi
done

if [ -z "$ONNX2NCNN" ]; then
    echo "❌ Утилита onnx2ncnn не найдена!"
    echo ""
    echo "Проверяю сборку NCNN..."
    if [ -d "$HOME/ncnn/build" ]; then
        echo "Сборка NCNN в процессе или завершена. Проверяю..."
        cd "$HOME/ncnn/build"
        if [ ! -f "tools/onnx/onnx2ncnn" ]; then
            echo "Собираю onnx2ncnn..."
            make tools/onnx/onnx2ncnn -j4 || make -j4
        fi
        if [ -f "tools/onnx/onnx2ncnn" ]; then
            ONNX2NCNN="$HOME/ncnn/build/tools/onnx/onnx2ncnn"
            echo "✓ Утилита найдена после сборки: $ONNX2NCNN"
        fi
    fi
fi

if [ -z "$ONNX2NCNN" ]; then
    echo ""
    echo "Для конвертации нужно собрать NCNN:"
    echo "  cd ~/ncnn && mkdir -p build && cd build"
    echo "  cmake .. -DCMAKE_BUILD_TYPE=Release -DNCNN_BUILD_TOOLS=ON"
    echo "  make -j4"
    exit 1
fi

# Конвертация
INPUT_MODEL="model/lp_regressor.onnx"
OUTPUT_DIR="model_ncnn"

if [ ! -f "$INPUT_MODEL" ]; then
    echo "❌ Модель не найдена: $INPUT_MODEL"
    exit 1
fi

echo ""
echo "Конвертация $INPUT_MODEL -> NCNN..."
mkdir -p "$OUTPUT_DIR"

MODEL_NAME="lp_regressor"
PARAM_FILE="$OUTPUT_DIR/${MODEL_NAME}.param"
BIN_FILE="$OUTPUT_DIR/${MODEL_NAME}.bin"

"$ONNX2NCNN" "$INPUT_MODEL" "$PARAM_FILE" "$BIN_FILE"

if [ -f "$PARAM_FILE" ] && [ -f "$BIN_FILE" ]; then
    echo ""
    echo "✓ Конвертация успешна!"
    echo "  Param: $PARAM_FILE ($(du -h "$PARAM_FILE" | cut -f1))"
    echo "  Bin:   $BIN_FILE ($(du -h "$BIN_FILE" | cut -f1))"
    echo ""
    
    # Тестирование производительности
    echo "Тестирование производительности..."
    echo ""
    echo "=== ONNX Runtime ==="
    python3 run_rpi.py --model full --threads 4 --input input --output output_onnx 2>&1 | grep -E "(Processing|Infer:|TOTAL:)" | head -6
    
    echo ""
    echo "=== NCNN ==="
    python3 run_rpi_ncnn.py --model "$OUTPUT_DIR" --threads 4 --input input --output output_ncnn 2>&1 | grep -E "(Processing|Infer:|TOTAL:)" | head -6
    
    echo ""
    echo "✓ Тестирование завершено!"
    echo "Результаты сохранены в output_onnx/ и output_ncnn/"
else
    echo "❌ Ошибка конвертации"
    exit 1
fi


