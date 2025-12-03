# Конвертация моделей

## ONNX → NCNN

NCNN формат может быть **в 2-3 раза быстрее** на Raspberry Pi по сравнению с ONNX Runtime.

### Шаг 1: Установка NCNN

#### Вариант A: Сборка из исходников (рекомендуется)

```bash
# Установка зависимостей
sudo apt-get update
sudo apt-get install build-essential cmake git

# Клонирование и сборка
git clone https://github.com/Tencent/ncnn.git
cd ncnn
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

# Утилита будет в: build/tools/onnx/onnx2ncnn
```

#### Вариант B: Python библиотека (проще)

```bash
pip install ncnn
```

**Примечание**: Python библиотека `ncnn` позволяет использовать NCNN модели, но для конвертации ONNX → NCNN всё ещё нужна утилита `onnx2ncnn` из варианта A.

### Шаг 2: Конвертация

```bash
# Базовая конвертация
python3 convert_to_ncnn.py --input model/lp_regressor.onnx --output model_ncnn

# С указанием пути к onnx2ncnn
python3 convert_to_ncnn.py \
    --input model/lp_regressor.onnx \
    --output model_ncnn \
    --onnx2ncnn ~/ncnn/build/tools/onnx/onnx2ncnn
```

### Шаг 3: Использование NCNN модели

```bash
python3 run_rpi_ncnn.py --model model_ncnn --threads 4
```

## ONNX → TensorFlow Lite (альтернатива)

TensorFlow Lite также хорошо работает на Raspberry Pi:

```bash
# Установка
pip install tf2onnx onnx-tf tensorflow

# Конвертация
python -m tf2onnx.convert \
    --onnx model/lp_regressor.onnx \
    --output model/lp_regressor.tflite \
    --opset 13
```

## Сравнение форматов

| Формат | Размер | Скорость на RPi | Установка |
|--------|--------|-----------------|-----------|
| ONNX   | 5.6MB  | ~100ms          | pip install onnxruntime |
| NCNN   | ~3MB   | ~30-50ms        | Требует сборки |
| TFLite | ~3MB   | ~60-80ms        | pip install tensorflow |

**Рекомендация**: Для максимальной производительности используйте NCNN.

