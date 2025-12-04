#!/usr/bin/env python3
"""
Тест NanoDet Light на 192x192 с профилированием этапов
"""
import time
import cv2
import numpy as np
import ncnn
import os
from pathlib import Path

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

class LightNanoDet:
    def __init__(self, param_path, bin_path, input_size=192):
        self.input_size = input_size
        
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = 4
        self.net.opt.use_fp16_packed = True
        self.net.opt.use_fp16_storage = True
        self.net.opt.use_packing_layout = True
        
        self.net.load_param(param_path)
        self.net.load_model(bin_path)
        
        # Имена выходов
        self.outputs = {
            8:  {'cls': 'cls_pred', 'reg': '1080'},
            16: {'cls': 'reg_pred', 'reg': '1107'},
            32: {'cls': '1132',     'reg': '1134'}
        }
        
        # Константы нормализации
        self.mean_vals = [103.53, 116.28, 123.675]
        self.norm_vals = [0.017429, 0.017507, 0.017125]

    def detect_profiled(self, img):
        # 1. Preprocess
        t0 = time.time()
        
        # Resize
        img_resized = cv2.resize(img, (self.input_size, self.input_size))
        t1 = time.time()
        
        # To NCNN Mat + Normalize
        mat_in = ncnn.Mat.from_pixels(img_resized, ncnn.Mat.PixelType.PIXEL_BGR, self.input_size, self.input_size)
        mat_in.substract_mean_normalize(self.mean_vals, self.norm_vals)
        t2 = time.time()
        
        # 2. Inference
        ex = self.net.create_extractor()
        ex.input("input", mat_in)
        
        results = {}
        for stride, names in self.outputs.items():
            ex.extract(names['cls']) # Просто дергаем, чтобы вычисления прошли
            ex.extract(names['reg'])
            
        t3 = time.time()
        
        return {
            'resize': (t1 - t0) * 1000,
            'prepare': (t2 - t1) * 1000,
            'inference': (t3 - t2) * 1000,
            'total': (t3 - t0) * 1000
        }

def benchmark():
    log("="*60)
    log("Бенчмарк NanoDet Light @ 192x192 (с профилированием)")
    log("="*60)
    
    param = "../export/nanodet_light.param"
    bin = "../export/nanodet_light.bin"
    
    detector = LightNanoDet(param, bin, input_size=192)
    
    # Загрузка фото
    input_dir = "../input"
    images = []
    for p in Path(input_dir).glob("*.jpg"):
        img = cv2.imread(str(p))
        if img is not None:
            images.append(img)
    images = images[:20]
    
    # Warmup
    log("Warmup...")
    for _ in range(20):
        detector.detect_profiled(images[0])
    
    # Test
    log("Testing...")
    stats = {'resize': [], 'prepare': [], 'inference': [], 'total': []}
    
    for img in images:
        for _ in range(5):
            res = detector.detect_profiled(img)
            for k, v in res.items():
                stats[k].append(v)
    
    # Результаты
    avg_total = np.mean(stats['total'])
    avg_infer = np.mean(stats['inference'])
    avg_pre = np.mean(stats['resize']) + np.mean(stats['prepare'])
    
    fps_total = 1000 / avg_total
    fps_infer = 1000 / avg_infer
    
    log("-" * 60)
    log(f"Разрешение: 192x192")
    log("-" * 60)
    log(f"Resize (cv2):      {np.mean(stats['resize']):.2f} ms")
    log(f"ToMat+Norm (NCNN): {np.mean(stats['prepare']):.2f} ms")
    log(f"Inference (NCNN):  {avg_infer:.2f} ms")
    log("-" * 60)
    log(f"TOTAL:             {avg_total:.2f} ms")
    log(f"FPS (Total):       {fps_total:.2f}")
    log(f"FPS (Infer only):  {fps_infer:.2f}")
    log("-" * 60)
    
    # Анализ
    pre_pct = (avg_pre / avg_total) * 100
    inf_pct = (avg_infer / avg_total) * 100
    log(f"Распределение времени:")
    log(f"  Предобработка: {pre_pct:.1f}%")
    log(f"  Инференс:      {inf_pct:.1f}%")

if __name__ == "__main__":
    benchmark()

