# Финальное сравнение производительности ONNX Runtime vs NCNN

## Оптимизации

### ONNX Runtime
- `intra_op_num_threads = 4`
- `execution_mode = ORT_PARALLEL`
- `enable_mem_pattern = True`
- `enable_cpu_mem_arena = True`
- `graph_optimization_level = ORT_ENABLE_ALL`

### NCNN
- `opt.num_threads = 4`
- `opt.use_vulkan_compute = False` (CPU only)
- `opt.use_fp16_packed = True`
- `opt.use_fp16_storage = True`
- `opt.use_fp16_arithmetic = True`
- `opt.use_packing_layout = True`
- `opt.use_winograd_convolution = True`
- `opt.lightmode = True`

## Результаты тестирования (200 изображений)

| Метрика | ONNX Runtime | NCNN |
|---------|--------------|------|
| **Среднее время** | **102.7 ms** | **102.4 ms** |
| **Мин. время** | 91.5 ms | **75.7 ms** |
| **Макс. время** | 170.8 ms | 311.7 ms |
| **Размер модели** | 5.7 MB | **2.9 MB** |

## Выводы

1. **Производительность идентична**: Оба фреймворка показывают одинаковую среднюю скорость (~102 ms).
2. **NCNN имеет меньший размер**: Модель занимает в 2 раза меньше места.
3. **NCNN может быть быстрее**: Минимальное время у NCNN лучше (75ms vs 91ms), что говорит о потенциале для дальнейшей оптимизации или зависимости от конкретных изображений.

## Итоговая рекомендация

- Используйте **NCNN**, если важен размер модели или есть ограничения по памяти.
- Используйте **ONNX Runtime**, если важна простота интеграции и стабильность времени выполнения.

Оба варианта отлично подходят для продакшена на Raspberry Pi 4.


