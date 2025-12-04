import os
import sys
import onnx
from onnxconverter_common import float16

def convert_fp16():
    print("\n=== КОНВЕРТАЦИЯ СУЩЕСТВУЮЩЕГО ONNX В FP16 ===")
    
    input_onnx = '/home/pi/Projects/nn_pro/rpi_deploy/export/nanodet_320_simplified.onnx'
    output_onnx = '/home/pi/Projects/nn_pro/rpi_deploy/export/nanodet_320_fp16.onnx'
    
    if not os.path.exists(input_onnx):
        print(f"Ошибка: файл {input_onnx} не найден")
        return

    try:
        print(f"Загрузка модели: {input_onnx}")
        model = onnx.load(input_onnx)
        
        print("Конвертация в FP16...")
        model_fp16 = float16.convert_float_to_float16(model)
        
        print(f"Сохранение: {output_onnx}")
        onnx.save(model_fp16, output_onnx)
        
        size_orig = os.path.getsize(input_onnx)/1e6
        size_fp16 = os.path.getsize(output_onnx)/1e6
        
        print(f"Готово!")
        print(f"Размер оригинала: {size_orig:.2f} MB")
        print(f"Размер FP16: {size_fp16:.2f} MB")
        print(f"Сжатие: {size_orig/size_fp16:.1f}x")
        
    except Exception as e:
        print(f"Ошибка конвертации: {e}")

if __name__ == "__main__":
    convert_fp16()

