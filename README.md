# Детектирование номерных знаков

Проект для детектирования номерных знаков на изображениях с использованием YOLO модели и ONNX Runtime.

## Установка

1. Создайте виртуальное окружение (рекомендуется для Raspberry Pi):
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Установите зависимости:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Важно:** При каждом запуске скриптов активируйте виртуальное окружение:
```bash
source venv/bin/activate
```

2. Скопируйте модель `best.onnx` в каталог проекта.

   Если модель находится на другой машине (например, на Mac), используйте один из способов:
   
   **Способ 1: Через SCP (если есть доступ по сети)**
   ```bash
   scp user@mac:/Users/dp/Projects/nn_park/best.onnx ./best.onnx
   ```
   
   **Способ 2: Через USB/внешний диск**
   ```bash
   # Подключите диск и скопируйте файл
   cp /media/usb/best.onnx ./best.onnx
   ```
   
   **Способ 3: Используйте скрипт (если модель доступна по указанному пути)**
   ```bash
   ./copy_model.sh
   ```

## Использование

**⚠️ ВАЖНО: Всегда активируйте виртуальное окружение перед запуском!**

**Вариант 1: С использованием скрипта запуска (САМЫЙ ПРОСТОЙ)**
```bash
./run_detection.sh --input ./input_images --output ./output_images --model best.onnx
```

**Вариант 2: С активацией виртуального окружения вручную**
```bash
source venv/bin/activate
python detect_plates.py --input ./input_images --output ./output_images --model best.onnx
```

**Вариант 3: Одной командой (без активации)**
```bash
venv/bin/python detect_plates.py --input ./input_images --output ./output_images --model best.onnx
```

### Параметры

- `--input` - Путь к входному каталогу с изображениями (обязательно)
- `--output` - Путь к выходному каталогу для результатов (обязательно)
- `--model` - Путь к ONNX модели (по умолчанию: `best.onnx`)
- `--conf` - Порог уверенности для детекций (по умолчанию: 0.25)
- `--iou` - Порог IoU для Non-Maximum Suppression (по умолчанию: 0.45)

### Пример

```bash
python detect_plates.py --input ./input_images --output ./output_images --model best.onnx --conf 0.3
```

## Структура проекта

- `detect_plates.py` - Основной скрипт для обработки изображений
- `yolo_postprocess.py` - Модуль для пост-обработки результатов YOLO модели (YOLO POS)
- `test_detection.py` - Тестовый скрипт для проверки на одном изображении
- `copy_model.sh` - Скрипт для копирования модели (если доступна по исходному пути)
- `best.onnx` - ONNX модель для детектирования (нужно скопировать)

## Тестирование

Для быстрой проверки на одном изображении:

```bash
python test_detection.py path/to/image.jpg best.onnx 0.25
```

## Особенности

- Обрабатывает все изображения из входного каталога (jpg, jpeg, png, bmp, tiff)
- Применяет YOLO post-processing с NMS для фильтрации дубликатов
- Выводит статистику обработки: скорость, количество детекций, время обработки
- Оптимизировано для работы на Raspberry Pi с использованием CPU

## Формат выходных изображений

Результаты сохраняются в выходной каталог с суффиксом `_detected`:
- `image.jpg` → `image_detected.jpg`
- На изображениях рисуются зеленые прямоугольники вокруг найденных номерных знаков
- Под каждым прямоугольником отображается уверенность детекции

