import torch
import torch.nn as nn
import sys
import os

# Читаем исходный код класса Model
with open('/home/pi/Projects/nn_pro/rpi_deploy/export/nanodet_320_simplified_pnnx.py', 'r') as f:
    code = f.read()

# Заменяем каналы (Backbone)
# Тяжелая (m): 24 -> [116 (58/58), 232 (116/116), 464 (232/232)]
# Легкая (t):  24 -> [48 (24/24), 96 (48/48), 192 (96/96)]

# Замены (строго в этом порядке чтобы не заменить лишнее)
# 58 -> 24
code = code.replace('in_channels=58', 'in_channels=24')
code = code.replace('out_channels=58', 'out_channels=24')
code = code.replace('groups=58', 'groups=24')

# 116 -> 48
code = code.replace('in_channels=116', 'in_channels=48')
code = code.replace('out_channels=116', 'out_channels=48')
code = code.replace('groups=116', 'groups=48')

# 232 -> 96
code = code.replace('in_channels=232', 'in_channels=96')
code = code.replace('out_channels=232', 'out_channels=96')
code = code.replace('groups=232', 'groups=96')

# 464 -> 192
code = code.replace('in_channels=464', 'in_channels=192')
code = code.replace('out_channels=464', 'out_channels=192')
code = code.replace('groups=464', 'groups=192')

# Head каналы (96) оставляем как есть, если они совпадают
# В NanoDet-t head каналы тоже 96.

# Сохраняем новый файл модели
with open('/home/pi/Projects/nn_pro/rpi_deploy/export/nanodet_t_model.py', 'w') as f:
    f.write(code)

print("Создан файл модели: export/nanodet_t_model.py")

# Теперь пытаемся загрузить веса и экспортировать
sys.path.append('/home/pi/Projects/nn_pro/rpi_deploy/export')
from nanodet_t_model import Model

def export():
    print("Создаем модель NanoDet-t...")
    model = Model()
    model.eval()
    
    ckpt_path = '/home/pi/Projects/nn_pro/rpi_deploy/export/last_v2.pth'
    print(f"Загружаем веса из {ckpt_path}...")
    
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt['model'] if 'model' in ckpt else ckpt
        
        # Маппинг имен
        # PNNX имена: conv2d_0, conv2d_1... (автосгенерированные по порядку выполнения)
        # PyTorch имена: backbone.conv1... (иерархические)
        
        # Проблема: мы не знаем порядок слоев в state_dict, чтобы сопоставить с conv2d_X
        # Но у нас есть порядок в коде Model.__init__!
        
        # Давайте попробуем "слепую" загрузку по порядку ключей (если порядок совпадает)
        # Получаем список модулей модели (они идут в порядке добавления в __init__)
        model_modules = [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.BatchNorm2d))]
        
        # Получаем список весов из state_dict (фильтруем num_batches_tracked)
        sd_keys = [k for k in state_dict.keys() if 'num_batches_tracked' not in k]
        
        # Это слишком рискованно.
        
        # Альтернатива: Попытаться переименовать ключи state_dict в имена Model.
        # Для этого нужно знать соответствие.
        
        print("❌ Невозможно автоматически сопоставить имена весов.")
        print("Нужен ONNX файл от пользователя.")
        
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    # export()
    pass

