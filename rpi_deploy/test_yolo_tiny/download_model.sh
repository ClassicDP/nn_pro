#!/bin/bash
# Скачивание YOLOv3-tiny модели

echo "Скачивание YOLOv3-tiny модели..."

# Конфиг
if [ ! -f "yolov3-tiny.cfg" ]; then
    echo "Скачивание конфига..."
    wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg
else
    echo "✓ Конфиг уже существует"
fi

# Веса
if [ ! -f "yolov3-tiny.weights" ]; then
    echo "Скачивание весов (33 MB)..."
    wget https://pjreddie.com/media/files/yolov3-tiny.weights
else
    echo "✓ Веса уже существуют"
fi

echo ""
echo "Проверка файлов:"
ls -lh yolov3-tiny.*

