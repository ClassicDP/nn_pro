#!/bin/bash
# Скачивание YOLOv8n и экспорт в ONNX с входом 256x256

echo "Скачивание YOLOv8n и экспорт в ONNX 256x256..."

# Проверка Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 не найден"
    exit 1
fi

# Установка ultralytics если нужно
python3 -c "import ultralytics" 2>/dev/null || {
    echo "Установка ultralytics..."
    pip install ultralytics
}

# Экспорт YOLOv8n в ONNX с входом 256x256
echo "Экспорт YOLOv8n в ONNX 256x256..."
python3 << 'EOF'
from ultralytics import YOLO
import os

# Загружаем YOLOv8n
model = YOLO('yolov8n.pt')

# Экспортируем в ONNX с входом 256x256
model.export(format='onnx', imgsz=256, simplify=True)

print("✓ Модель экспортирована: yolov8n.onnx (256x256)")
EOF

if [ -f "yolov8n.onnx" ]; then
    echo "✓ Готово! Модель: yolov8n.onnx"
    ls -lh yolov8n.onnx
else
    echo "❌ Ошибка экспорта"
    exit 1
fi

