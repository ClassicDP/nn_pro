#!/usr/bin/env python3
"""
Бенчмарк для nanodet_light (NanoDet-t)
"""
import time
import cv2
import numpy as np
import ncnn
import os
from pathlib import Path
import random

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

class LightNanoDet:
    def __init__(self, param_path, bin_path, input_size=320, conf_threshold=0.25):
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = 4
        self.net.opt.use_fp16_packed = True
        self.net.opt.use_fp16_storage = True
        self.net.opt.use_packing_layout = True
        
        self.net.load_param(param_path)
        self.net.load_model(bin_path)
        
        # Имена выходов согласно nanodet_light.param
        self.outputs = {
            8:  {'cls': 'cls_pred', 'reg': '1080'},
            16: {'cls': 'reg_pred', 'reg': '1107'},  # reg_pred это имя cls блоба для stride 16!
            32: {'cls': '1132',     'reg': '1134'}
        }
        
        log(f"Модель загружена: {param_path}")

    def preprocess(self, img):
        img_resized = cv2.resize(img, (self.input_size, self.input_size))
        # NanoDet требует BGR (cv2 default) или RGB? Обычно RGB для NCNN models
        # Но если модель обучалась в PyTorch/NanoDet репо, там обычно используется:
        # mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395] (BGR)
        # Проверим стандартные значения
        
        mean_vals = [103.53, 116.28, 123.675]
        norm_vals = [0.017429, 0.017507, 0.017125] # 1/57.375...
        
        mat_in = ncnn.Mat.from_pixels(img_resized, ncnn.Mat.PixelType.PIXEL_BGR, self.input_size, self.input_size)
        mat_in.substract_mean_normalize(mean_vals, norm_vals)
        return mat_in

    def detect(self, img):
        mat_in = self.preprocess(img)
        ex = self.net.create_extractor()
        ex.input("input", mat_in)
        
        results = {}
        for stride, names in self.outputs.items():
            # Class
            ret, cls_blob = ex.extract(names['cls'])
            if ret == 0: results[f'cls_{stride}'] = np.array(cls_blob)
            
            # Reg
            ret, reg_blob = ex.extract(names['reg'])
            if ret == 0: results[f'reg_{stride}'] = np.array(reg_blob)
            
        return results

def benchmark():
    log("="*60)
    log("Бенчмарк NanoDet Light (1.2 MB)")
    log("="*60)
    
    param = "../export/nanodet_light.param"
    bin = "../export/nanodet_light.bin"
    
    if not os.path.exists(param) or not os.path.exists(bin):
        log(f"Файлы не найдены!")
        return

    # Загрузка фото
    input_dir = "../input"
    images = []
    for p in Path(input_dir).glob("*.jpg"):
        img = cv2.imread(str(p))
        if img is not None:
            images.append(img)
    images = images[:20]
    log(f"Загружено {len(images)} изображений")
    
    detector = LightNanoDet(param, bin)
    
    # Warmup
    log("Warmup...")
    for _ in range(20):
        detector.detect(images[0])
    
    # Test
    log("Testing...")
    times = []
    for img in images:
        for _ in range(3): # 3 прогона на каждое фото
            t0 = time.time()
            res = detector.detect(img)
            times.append((time.time() - t0) * 1000)
    
    avg = np.mean(times)
    fps = 1000 / avg
    
    log("-" * 60)
    log(f"Среднее время: {avg:.2f} ms")
    log(f"FPS: {fps:.2f}")
    log(f"Мин: {np.min(times):.2f} ms")
    log(f"Макс: {np.max(times):.2f} ms")
    log("-" * 60)
    
    # Сравнение
    log("Сравнение с предыдущими результатами:")
    log(f"Heavy Model (3.6 MB): ~100 ms (10 FPS)")
    log(f"Model Zoo (1.9 MB):   ~50 ms (20 FPS)")
    log(f"Light Model (1.2 MB): ~{avg:.1f} ms ({fps:.1f} FPS)")
    
    if avg < 25:
        log("\n🚀 ОТЛИЧНЫЙ РЕЗУЛЬТАТ! Подходит для реалтайма (>40 FPS)")
    elif avg < 40:
        log("\n✅ ХОРОШИЙ РЕЗУЛЬТАТ! Подходит для реалтайма (25-30 FPS)")

if __name__ == "__main__":
    benchmark()

