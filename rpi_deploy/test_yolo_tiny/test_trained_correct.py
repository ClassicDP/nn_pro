#!/usr/bin/env python3
"""
Правильный тест обученной NanoDet модели:
- Один ресайз до 320x320 (нативный размер модели)
- Сравнение ONNX Runtime vs Model Zoo (NCNN)
"""
import cv2
import numpy as np
import time
import os
import sys
import random
import onnxruntime as ort
from pathlib import Path

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()

class TrainedNanoDetONNX:
    """Детектор на ONNX Runtime - один ресайз до 320x320"""
    
    def __init__(self, onnx_path, conf_threshold=0.35, num_threads=4):
        self.input_size = 320  # Нативный размер модели
        self.conf_threshold = conf_threshold
        
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = num_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            onnx_path, sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        log(f"✓ ONNX модель загружена: {onnx_path}")
        log(f"  - Вход: {self.input_name}, размер: 320x320")
        log(f"  - Выходы: {self.output_names}")
    
    def preprocess(self, img):
        """Один ресайз до 320x320"""
        orig_h, orig_w = img.shape[:2]
        
        # Простой ресайз до 320x320
        img_resized = cv2.resize(img, (self.input_size, self.input_size))
        
        # BGR -> RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Нормализация ImageNet
        img_float = img_rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_float - mean) / std
        
        # [1, 3, H, W]
        img_tensor = np.transpose(img_norm, (2, 0, 1))
        img_tensor = np.expand_dims(img_tensor, axis=0)
        
        return img_tensor, (orig_h, orig_w)
    
    def decode_predictions(self, outputs, orig_shape):
        """Декодирование предсказаний"""
        detections = []
        orig_h, orig_w = orig_shape
        strides = [8, 16, 32]
        
        # Разделяем выходы на cls и reg
        cls_outputs = []
        reg_outputs = []
        
        for out_name in sorted(self.output_names):
            output = outputs[out_name]
            if 'cls' in out_name.lower():
                cls_outputs.append(output)
            elif 'reg' in out_name.lower():
                reg_outputs.append(output)
        
        # Сортируем по stride
        cls_by_stride = {}
        reg_by_stride = {}
        for out_name in self.output_names:
            output = outputs[out_name]
            for s in strides:
                if f'stride{s}' in out_name:
                    if 'cls' in out_name.lower():
                        cls_by_stride[s] = output
                    elif 'reg' in out_name.lower():
                        reg_by_stride[s] = output
        
        for stride in strides:
            if stride not in cls_by_stride or stride not in reg_by_stride:
                continue
                
            cls_pred = cls_by_stride[stride]
            reg_pred = reg_by_stride[stride]
            
            _, c, h, w = cls_pred.shape
            
            # Sigmoid
            scores = 1.0 / (1.0 + np.exp(-cls_pred))
            
            for i in range(h):
                for j in range(w):
                    score = scores[0, 0, i, j]
                    
                    if score < self.conf_threshold:
                        continue
                    
                    # Декодирование bbox
                    l = reg_pred[0, 0, i, j]
                    t = reg_pred[0, 1, i, j]
                    r = reg_pred[0, 2, i, j]
                    b = reg_pred[0, 3, i, j]
                    
                    cx = (j + 0.5) * stride
                    cy = (i + 0.5) * stride
                    
                    x1 = (cx - l) / self.input_size * orig_w
                    y1 = (cy - t) / self.input_size * orig_h
                    x2 = (cx + r) / self.input_size * orig_w
                    y2 = (cy + b) / self.input_size * orig_h
                    
                    x1 = max(0, min(orig_w, x1))
                    y1 = max(0, min(orig_h, y1))
                    x2 = max(0, min(orig_w, x2))
                    y2 = max(0, min(orig_h, y2))
                    
                    if x2 > x1 and y2 > y1:
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'score': float(score),
                            'class': 0
                        })
        
        return self.nms(detections)
    
    def nms(self, detections, iou_threshold=0.5):
        if len(detections) == 0:
            return []
        
        detections = sorted(detections, key=lambda x: x['score'], reverse=True)
        keep = []
        
        while detections:
            best = detections.pop(0)
            keep.append(best)
            detections = [d for d in detections 
                         if self.compute_iou(best['bbox'], d['bbox']) < iou_threshold]
        
        return keep
    
    def compute_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
        xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
        
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0
    
    def detect(self, img):
        img_tensor, orig_shape = self.preprocess(img)
        outputs = self.session.run(self.output_names, {self.input_name: img_tensor})
        outputs_dict = {name: out for name, out in zip(self.output_names, outputs)}
        return self.decode_predictions(outputs_dict, orig_shape)

def main():
    log("="*70)
    log("Сравнение: Обученная модель (ONNX) vs Model Zoo (NCNN)")
    log("="*70)
    
    # Параметры
    model_path = "../export/nanodet_320_simplified.onnx"
    if not os.path.exists(model_path):
        model_path = "../export/nanodet_320.onnx"
    
    input_dir = "../input"
    num_images = 20
    conf_threshold = 0.25  # Снижаем для сравнения
    
    # Находим изображения
    image_files = list(Path(input_dir).glob('*.jpg'))
    random.seed(42)
    selected = random.sample(image_files, min(num_images, len(image_files)))
    
    # Предзагрузка
    images_data = []
    for p in selected:
        img = cv2.imread(str(p))
        if img is not None:
            images_data.append((p, img))
    
    log(f"Загружено {len(images_data)} изображений")
    
    # === ТЕСТ 1: Обученная модель через ONNX Runtime ===
    log("\n" + "="*70)
    log("ТЕСТ 1: Обученная модель (ONNX Runtime)")
    log("="*70)
    
    try:
        detector_onnx = TrainedNanoDetONNX(model_path, conf_threshold=conf_threshold)
        
        # Warmup
        for _ in range(3):
            detector_onnx.detect(images_data[0][1])
        
        onnx_times = []
        onnx_found = 0
        onnx_detections = 0
        
        for img_path, img in images_data:
            t0 = time.time()
            dets = detector_onnx.detect(img)
            onnx_times.append((time.time() - t0) * 1000)
            onnx_detections += len(dets)
            if dets:
                onnx_found += 1
        
        avg_onnx = np.mean(onnx_times)
        log(f"  Среднее время: {avg_onnx:.1f} ms")
        log(f"  FPS: {1000/avg_onnx:.1f}")
        log(f"  Найдено: {onnx_found}/{len(images_data)} ({100*onnx_found/len(images_data):.1f}%)")
        log(f"  Всего детекций: {onnx_detections}")
    except Exception as e:
        log(f"❌ Ошибка ONNX: {e}")
        import traceback
        traceback.print_exc()
    
    # === ТЕСТ 2: Model Zoo NanoDet (NCNN) ===
    log("\n" + "="*70)
    log("ТЕСТ 2: Model Zoo NanoDet (NCNN)")
    log("="*70)
    
    try:
        from ncnn.model_zoo import get_model
        
        # COCO vehicle classes
        VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorbike, bus, truck
        
        detector_zoo = get_model("nanodet", target_size=320, prob_threshold=conf_threshold, num_threads=4)
        log(f"✓ Model Zoo NanoDet загружен (NCNN, 320x320)")
        
        # Warmup
        for _ in range(3):
            detector_zoo(images_data[0][1])
        
        zoo_times = []
        zoo_found = 0
        zoo_detections = 0
        
        for img_path, img in images_data:
            t0 = time.time()
            objects = detector_zoo(img)
            zoo_times.append((time.time() - t0) * 1000)
            
            # Фильтруем транспорт
            vehicles = [o for o in objects if o.label in VEHICLE_CLASSES]
            zoo_detections += len(vehicles)
            if vehicles:
                zoo_found += 1
        
        avg_zoo = np.mean(zoo_times)
        log(f"  Среднее время: {avg_zoo:.1f} ms")
        log(f"  FPS: {1000/avg_zoo:.1f}")
        log(f"  Найдено: {zoo_found}/{len(images_data)} ({100*zoo_found/len(images_data):.1f}%)")
        log(f"  Всего детекций: {zoo_detections}")
    except Exception as e:
        log(f"❌ Ошибка Model Zoo: {e}")
        import traceback
        traceback.print_exc()
    
    # === СРАВНЕНИЕ ===
    log("\n" + "="*70)
    log("СРАВНЕНИЕ")
    log("="*70)
    log(f"{'Модель':<30} {'Время (ms)':<15} {'FPS':<10} {'Найдено %':<15} {'Детекций':<10}")
    log("-" * 70)
    
    if 'avg_onnx' in dir():
        log(f"{'Обученная (ONNX Runtime)':<30} {avg_onnx:>6.1f} ms      {1000/avg_onnx:>5.1f}     {100*onnx_found/len(images_data):>5.1f}%          {onnx_detections}")
    
    if 'avg_zoo' in dir():
        log(f"{'Model Zoo (NCNN)':<30} {avg_zoo:>6.1f} ms      {1000/avg_zoo:>5.1f}     {100*zoo_found/len(images_data):>5.1f}%          {zoo_detections}")
    
    if 'avg_onnx' in dir() and 'avg_zoo' in dir():
        speedup = avg_onnx / avg_zoo
        log(f"\n⚡ NCNN быстрее в {speedup:.1f}x раз")
        log(f"\nВывод: Нужно конвертировать обученную модель в NCNN для ускорения!")

if __name__ == "__main__":
    main()

