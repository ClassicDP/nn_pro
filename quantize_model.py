"""
Динамическая квантизация ONNX модели в INT8
"""
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

input_model = "best.onnx"
output_model = "best_int8.onnx"

# Проверяем размер исходной модели
original_size = os.path.getsize(input_model) / (1024 * 1024)
print(f"Исходная модель: {input_model} ({original_size:.2f} MB)")

# Динамическая квантизация в INT8
print("Выполняется квантизация...")
quantize_dynamic(
    model_input=input_model,
    model_output=output_model,
    weight_type=QuantType.QUInt8
)

# Проверяем размер квантизированной модели
quantized_size = os.path.getsize(output_model) / (1024 * 1024)
print(f"Квантизированная модель: {output_model} ({quantized_size:.2f} MB)")
print(f"Сжатие: {original_size/quantized_size:.1f}x")
print("Готово!")





