# Быстрый старт

## Как запустить обработку изображений

### Способ 1: Использовать готовый скрипт (РЕКОМЕНДУЕТСЯ)
```bash
./run_detection.sh --input ./input_images --output ./output_images --model best.onnx
```

### Способ 2: Активировать виртуальное окружение вручную
```bash
# Шаг 1: Активировать виртуальное окружение
source venv/bin/activate

# Шаг 2: Запустить скрипт
python detect_plates.py --input ./input_images --output ./output_images --model best.onnx
```

### Способ 3: Без активации (прямой вызов)
```bash
venv/bin/python detect_plates.py --input ./input_images --output ./output_images --model best.onnx
```

## Проверка работы

Проверить на одном изображении:
```bash
source venv/bin/activate
python test_detection.py input_images/ваше_изображение.jpg best.onnx 0.25
```

## Параметры

- `--input` - папка с входными изображениями (обязательно)
- `--output` - папка для сохранения результатов (обязательно)
- `--model` - путь к модели ONNX (по умолчанию: best.onnx)
- `--conf` - порог уверенности (по умолчанию: 0.25)
- `--iou` - порог IoU для NMS (по умолчанию: 0.45)

## Примеры

Обработка с низким порогом уверенности (найдет больше детекций):
```bash
./run_detection.sh --input ./input_images --output ./output_images --model best.onnx --conf 0.15
```

Обработка с высоким порогом уверенности (только очень уверенные детекции):
```bash
./run_detection.sh --input ./input_images --output ./output_images --model best.onnx --conf 0.5
```






