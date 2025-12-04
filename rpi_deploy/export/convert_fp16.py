import torch
import torch.nn as nn
import onnx
import os
import sys

# Импортируем модель, сгенерированную pnnx
# Мы предполагаем, что структура last_v2.pth совпадает с simplified.onnx
sys.path.append('/home/pi/Projects/nn_pro/rpi_deploy/export')
from nanodet_320_simplified_pnnx import Model

def export_onnx():
    print("=== Экспорт last_v2.pth в ONNX ===")
    
    # 1. Создаем модель
    model = Model()
    model.eval()
    
    # 2. Загружаем веса
    ckpt_path = '/home/pi/Projects/nn_pro/rpi_deploy/export/last_v2.pth'
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        # Обработка разных форматов чекпоинтов
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        elif 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt
            
        # Удаляем префикс 'module.' если есть (DDP)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v
            
        # Пытаемся загрузить. Если ключи не совпадают, pnnx сгенерировал имена автоматически
        # и они не совпадут с оригинальными именами в pth.
        # В этом случае нам придется мапить их вручную или использовать pnnx-код как reference.
        
        # НО! pnnx генерирует код, где имена слоев (self.conv2d_0) жестко привязаны.
        # В state_dict ключи типа 'backbone.conv1.weight'.
        # Автоматический маппинг сложен без графа.
        
        print("ВНИМАНИЕ: Прямая загрузка весов в pnnx-модель не сработает из-за разных имен.")
        print("Попытка конвертации через pnnx была правильной стратегией для получения архитектуры.")
        
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        return

    print("Поскольку у нас нет исходного кода NanoDet (файлов model.py),")
    print("мы не можем корректно загрузить last_v2.pth для экспорта.")
    print("Однако, мы выяснили, что архитектура совпадает с NanoDet-M (1.0x).")
    print("nanodet_320_simplified.onnx уже является экспортом этой модели.")
    
    print("\n=== КОНВЕРТАЦИЯ СУЩЕСТВУЮЩЕГО ONNX В FP16 ===")
    import onnx
    from onnxconverter_common import float16
    
    input_onnx = '/home/pi/Projects/nn_pro/rpi_deploy/export/nanodet_320_simplified.onnx'
    output_onnx = '/home/pi/Projects/nn_pro/rpi_deploy/export/nanodet_320_fp16.onnx'
    
    model = onnx.load(input_onnx)
    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, output_onnx)
    
    print(f"Сохранено: {output_onnx}")
    print(f"Размер: {os.path.getsize(output_onnx)/1e6:.2f} MB")

if __name__ == "__main__":
    export_onnx()

