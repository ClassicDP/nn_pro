# Выбор модели

Проект поддерживает два типа моделей:

## ONNX модель (best.onnx)
- Стандартный формат
- Работает из коробки
- Размер: ~12MB

## NCNN модель (best-opt.bin + best-opt.param)
- Оптимизированная 8-битная модель
- **Быстрее на Raspberry Pi** (в 2-3 раза)
- Размер: ~6MB
- Требует установки NCNN: `pip install pyncnn`

## Использование

### ONNX модель:
```bash
./run_detection.sh --input ./input_images --output ./output_images --model best.onnx
```

### NCNN модель:
```bash
./run_detection.sh --input ./input_images --output ./output_images --model best-opt.bin
```

Модель выбирается автоматически по расширению файла:
- `.onnx` → ONNX Runtime
- `.bin` или `.param` → NCNN

## Установка NCNN

```bash
source venv/bin/activate
pip install pyncnn
```

Или соберите из исходников для лучшей производительности:
https://github.com/Tencent/ncnn
