#!/bin/bash
# Скрипт для копирования модели best.onnx

SOURCE_MODEL="/Users/dp/Projects/nn_park/best.onnx"
DEST_MODEL="./best.onnx"

if [ -f "$SOURCE_MODEL" ]; then
    echo "Копирование модели из $SOURCE_MODEL..."
    cp "$SOURCE_MODEL" "$DEST_MODEL"
    echo "Модель успешно скопирована в $DEST_MODEL"
else
    echo "Предупреждение: Модель не найдена по пути $SOURCE_MODEL"
    echo "Пожалуйста, скопируйте модель best.onnx вручную в каталог проекта"
fi

