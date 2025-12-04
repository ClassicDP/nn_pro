#!/bin/bash
# Автоматическая конвертация и тестирование когда сборка завершится

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Автоматическая конвертация ONNX -> NCNN ==="
echo ""

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
)

echo "Поиск утилиты onnx2ncnn..."
for path in "${POSSIBLE_PATHS[@]}"; do
    if [ -f "$path" ] && [ -x "$path" ]; then
        ONNX2NCNN="$path"
        echo "✓ Найдена: $ONNX2NCNN"
        break
    fi
done

if [ -z "$ONNX2NCNN" ]; then
    echo "⏳ Утилита не найдена. Ожидание завершения сборки NCNN..."
    echo "   (Это может занять 30-60 минут на Raspberry Pi)"
    echo ""
    
    # Ждем завершения сборки
    while [ ! -f "$HOME/ncnn/build/tools/onnx/onnx2ncnn" ]; do
        sleep 30
        if [ -d "$HOME/ncnn/build" ]; then
            echo "$(date +%H:%M:%S) - Ожидание..."
        else
            echo "Ошибка: Директория сборки не найдена!"
            exit 1
        fi
    done
    
    ONNX2NCNN="$HOME/ncnn/build/tools/onnx/onnx2ncnn"
    echo "✓ Утилита готова: $ONNX2NCNN"
fi

# Конвертация
INPUT_MODEL="model/lp_regressor.onnx"
OUTPUT_DIR="model_ncnn"

if [ ! -f "$INPUT_MODEL" ]; then
    echo "❌ Модель не найдена: $INPUT_MODEL"
    exit 1
fi

echo ""
echo "=== Конвертация ==="
mkdir -p "$OUTPUT_DIR"

MODEL_NAME="lp_regressor"
PARAM_FILE="$OUTPUT_DIR/${MODEL_NAME}.param"
BIN_FILE="$OUTPUT_DIR/${MODEL_NAME}.bin"

echo "Конвертация: $INPUT_MODEL -> NCNN..."
"$ONNX2NCNN" "$INPUT_MODEL" "$PARAM_FILE" "$BIN_FILE"

if [ -f "$PARAM_FILE" ] && [ -f "$BIN_FILE" ]; then
    echo ""
    echo "✓ Конвертация успешна!"
    echo "  Param: $PARAM_FILE ($(du -h "$PARAM_FILE" | cut -f1))"
    echo "  Bin:   $BIN_FILE ($(du -h "$BIN_FILE" | cut -f1))"
    echo ""
    
    # Тестирование производительности
    echo "=== Тестирование производительности ==="
    echo ""
    
    echo "1. ONNX Runtime:"
    python3 run_rpi.py --model full --threads 4 --input input --output output_onnx 2>&1 | grep -E "(Processing|Infer:|TOTAL:)" | head -6
    
    echo ""
    echo "2. NCNN:"
    python3 run_rpi_ncnn.py --model "$OUTPUT_DIR" --threads 4 --input input --output output_ncnn 2>&1 | grep -E "(Processing|Infer:|TOTAL:)" | head -6
    
    echo ""
    echo "✓✓✓ Тестирование завершено!"
    echo ""
    echo "Результаты сохранены в:"
    echo "  - ONNX: output_onnx/"
    echo "  - NCNN: output_ncnn/"
else
    echo "❌ Ошибка конвертации"
    exit 1
fi


