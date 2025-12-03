# License Plate Corner Detector для Raspberry Pi

Оптимизированный детектор углов номерных знаков для Raspberry Pi.

Поддерживает два формата моделей:
- **ONNX** - стандартный формат (по умолчанию)
- **NCNN** - быстрее на Raspberry Pi в 2-3 раза (требует конвертации)

## Быстрый старт

### ONNX модель (по умолчанию):
```bash
./run_rpi.sh
```

### NCNN модель (быстрее):
```bash
# 1. Установите NCNN библиотеку
pip install ncnn

# 2. Конвертируйте ONNX в NCNN (нужна утилита onnx2ncnn)
#    См. CONVERSION.md для инструкций по установке
python3 convert_to_ncnn.py --input model/lp_regressor.onnx --output model_ncnn

# 3. Запустите с NCNN
python3 run_rpi_ncnn.py --model model_ncnn --threads 4
```

## Переключение между моделями

### Использование пресетов:

```bash
# Обычная модель (рекомендуется - быстрее)
python3 run_rpi.py --model full --threads 4

# Квантованная модель (меньший размер, но медленнее)
python3 run_rpi.py --model quant --threads 4
```

### Использование своего пути:

```bash
python3 run_rpi.py --model /path/to/your/model.onnx --threads 4
```

## Параметры

- `--input` - директория с входными изображениями (по умолчанию: `input`)
- `--output` - директория для результатов (по умолчанию: `output`)
- `--model` - модель: `full`, `quant`, или путь к файлу
- `--threads` - количество потоков (по умолчанию: 4 для RPi 4)

## Производительность

### Обычная модель (`lp_regressor.onnx`):
- Размер: 5.6MB
- Инференция: ~100-110ms
- Рекомендуется для лучшей производительности

### Квантованная модель (`lp_regressor_quant.onnx`):
- Размер: 1.6MB
- Инференция: ~230-270ms
- Используйте если важнее размер модели

## Оптимизации

- Параллельный режим ONNX Runtime
- Оптимизация памяти
- Оптимизация графа
- Быстрая предобработка изображений

## Конвертация в NCNN

NCNN может быть **в 2-3 раза быстрее** на Raspberry Pi:

```bash
# Конвертация ONNX -> NCNN
python3 convert_to_ncnn.py --input model/lp_regressor.onnx --output model_ncnn

# Требуется установка NCNN:
# 1. Скачайте: https://github.com/Tencent/ncnn
# 2. Соберите: cd ncnn && mkdir build && cd build && cmake .. && make -j4
# 3. Утилита будет в: build/tools/onnx/onnx2ncnn
# 4. Установите Python библиотеку: pip install pyncnn
```

## Примеры

### ONNX модели:
```bash
# Обработка с обычной моделью
python3 run_rpi.py --model full --input my_images --output results

# Обработка с квантованной моделью (меньше памяти)
python3 run_rpi.py --model quant --threads 2

# Использование своей модели
python3 run_rpi.py --model ../best.onnx --threads 4
```

### NCNN модели:
```bash
# После конвертации используйте run_rpi_ncnn.py
python3 run_rpi_ncnn.py --model model_ncnn --threads 4
```

