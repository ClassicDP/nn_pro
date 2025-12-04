#!/usr/bin/env python3
"""
Тест обученной NanoDet модели через NCNN (конвертированной через pnnx)
Сравнение всех вариантов: ONNX Runtime, Model Zoo NCNN, Обученная NCNN
"""
import cv2
import numpy as np
import time
import os
import sys
import random
import ncnn
from pathlib import Path

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()

class TrainedNanoDetNCNN:
    """Обученная модель через NCNN"""
    
    def __init__(self, param_path, bin_path, input_size=320, conf_threshold=0.25, num_threads=4):
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        
        # Инициализация NCNN
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = num_threads
        self.net.opt.use_fp16_packed = True
        self.net.opt.use_fp16_storage = True
        self.net.opt.use_packing_layout = True
        
        # Загрузка модели
        ret_param = self.net.load_param(param_path)
        ret_model = self.net.load_model(bin_path)
        
        if ret_param != 0 or ret_model != 0:
            raise RuntimeError(f"Failed to load model: param={ret_param}, model={ret_model}")
        
        log(f"✓ NCNN модель загружена: {param_path}")
        log(f"  - Размер: {input_size}x{input_size}")
        log(f"  - Потоков: {num_threads}")
        
        # Получаем имена слоев (pnnx использует in0/out0-out5)
        self.input_name = "in0"
        # out0=cls_stride8, out1=cls_stride16, out2=cls_stride32
        # out3=reg_stride8, out4=reg_stride16, out5=reg_stride32
        self.output_mapping = {
            8: {'cls': 'out0', 'reg': 'out3'},
            16: {'cls': 'out1', 'reg': 'out4'},
            32: {'cls': 'out2', 'reg': 'out5'}
        }
    
    def preprocess(self, img):
        """Предобработка"""
        orig_h, orig_w = img.shape[:2]
        
        # Resize до 320x320
        img_resized = cv2.resize(img, (self.input_size, self.input_size))
        
        # BGR -> RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Создание ncnn Mat
        mat_in = ncnn.Mat.from_pixels(
            img_rgb, 
            ncnn.Mat.PixelType.PIXEL_RGB, 
            self.input_size, 
            self.input_size
        )
        
        # Нормализация ImageNet
        mean_vals = [123.675, 116.28, 103.53]  # 0.485*255, 0.456*255, 0.406*255
        norm_vals = [0.01712475, 0.0175, 0.01742919]  # 1/(0.229*255), ...
        mat_in.substract_mean_normalize(mean_vals, norm_vals)
        
        return mat_in, (orig_h, orig_w)
    
    def detect(self, img):
        """Детекция"""
        mat_in, orig_shape = self.preprocess(img)
        orig_h, orig_w = orig_shape
        
        # Инференс
        ex = self.net.create_extractor()
        ex.input(self.input_name, mat_in)
        
        # Извлекаем выходы
        outputs = {}
        for stride, names in self.output_mapping.items():
            for key in ['cls', 'reg']:
                name = names[key]
                ret, mat_out = ex.extract(name)
                if ret == 0:
                    outputs[f"{key}_{stride}"] = np.array(mat_out)
        
        # Декодируем
        detections = []
        strides = [8, 16, 32]
        
        for stride in strides:
            cls_key = f"cls_{stride}"
            reg_key = f"reg_{stride}"
            
            if cls_key not in outputs or reg_key not in outputs:
                continue
            
            cls_pred = outputs[cls_key]
            reg_pred = outputs[reg_key]
            
            # Определяем размеры
            if len(cls_pred.shape) == 3:
                c, h, w = cls_pred.shape
                cls_pred = cls_pred.reshape(c, h, w)
                reg_pred = reg_pred.reshape(4, h, w)
            else:
                _, c, h, w = cls_pred.shape
                cls_pred = cls_pred[0]
                reg_pred = reg_pred[0]
            
            # Sigmoid (vectorized)
            scores = 1.0 / (1.0 + np.exp(-cls_pred[0]))
            
            # Находим индексы где score > threshold (vectorized)
            mask = scores > self.conf_threshold
            if not np.any(mask):
                continue
            
            # Получаем координаты ячеек
            yi, xi = np.where(mask)
            valid_scores = scores[mask]
            
            # Центры ячеек
            cx = (xi + 0.5) * stride
            cy = (yi + 0.5) * stride
            
            # Регрессия
            l = reg_pred[0][mask]
            t = reg_pred[1][mask]
            r = reg_pred[2][mask]
            b = reg_pred[3][mask]
            
            # Координаты bbox
            x1 = (cx - l) / self.input_size * orig_w
            y1 = (cy - t) / self.input_size * orig_h
            x2 = (cx + r) / self.input_size * orig_w
            y2 = (cy + b) / self.input_size * orig_h
            
            # Clip
            x1 = np.clip(x1, 0, orig_w)
            y1 = np.clip(y1, 0, orig_h)
            x2 = np.clip(x2, 0, orig_w)
            y2 = np.clip(y2, 0, orig_h)
            
            # Фильтруем валидные bbox
            valid = (x2 > x1) & (y2 > y1)
            
            for i in range(len(valid_scores)):
                if valid[i]:
                    detections.append({
                        'bbox': [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
                        'score': float(valid_scores[i]),
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

def main():
    log("="*70)
    log("Полное сравнение: ONNX Runtime vs Model Zoo NCNN vs Обученная NCNN")
    log("="*70)
    
    # Пути
    ncnn_param = "../export/nanodet_320_simplified.ncnn.param"
    ncnn_bin = "../export/nanodet_320_simplified.ncnn.bin"
    onnx_path = "../export/nanodet_320_simplified.onnx"
    input_dir = "../input"
    num_images = 20
    conf_threshold = 0.25
    
    # Загрузка изображений
    image_files = list(Path(input_dir).glob('*.jpg'))
    random.seed(42)
    selected = random.sample(image_files, min(num_images, len(image_files)))
    
    images_data = []
    for p in selected:
        img = cv2.imread(str(p))
        if img is not None:
            images_data.append((p, img))
    
    log(f"Загружено {len(images_data)} изображений")
    
    results = {}
    
    # === ТЕСТ 1: Обученная NCNN ===
    log("\n" + "="*70)
    log("ТЕСТ 1: Обученная модель (NCNN)")
    log("="*70)
    
    try:
        detector = TrainedNanoDetNCNN(ncnn_param, ncnn_bin, conf_threshold=conf_threshold)
        
        # Warmup
        for _ in range(5):
            detector.detect(images_data[0][1])
        
        times = []
        found = 0
        total_dets = 0
        
        for img_path, img in images_data:
            t0 = time.time()
            dets = detector.detect(img)
            times.append((time.time() - t0) * 1000)
            total_dets += len(dets)
            if dets:
                found += 1
        
        avg = np.mean(times)
        results['trained_ncnn'] = {
            'time': avg,
            'fps': 1000/avg,
            'found': found,
            'detections': total_dets
        }
        
        log(f"  Среднее время: {avg:.1f} ms")
        log(f"  FPS: {1000/avg:.1f}")
        log(f"  Найдено: {found}/{len(images_data)} ({100*found/len(images_data):.1f}%)")
        log(f"  Всего детекций: {total_dets}")
    except Exception as e:
        log(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
    
    # === ТЕСТ 2: Model Zoo NCNN ===
    log("\n" + "="*70)
    log("ТЕСТ 2: Model Zoo NanoDet (NCNN)")
    log("="*70)
    
    try:
        from ncnn.model_zoo import get_model
        
        VEHICLE_CLASSES = {2, 3, 5, 7}
        detector = get_model("nanodet", target_size=320, prob_threshold=conf_threshold, num_threads=4)
        log(f"✓ Model Zoo NanoDet загружен")
        
        # Warmup
        for _ in range(5):
            detector(images_data[0][1])
        
        times = []
        found = 0
        total_dets = 0
        
        for img_path, img in images_data:
            t0 = time.time()
            objects = detector(img)
            times.append((time.time() - t0) * 1000)
            
            vehicles = [o for o in objects if o.label in VEHICLE_CLASSES]
            total_dets += len(vehicles)
            if vehicles:
                found += 1
        
        avg = np.mean(times)
        results['model_zoo'] = {
            'time': avg,
            'fps': 1000/avg,
            'found': found,
            'detections': total_dets
        }
        
        log(f"  Среднее время: {avg:.1f} ms")
        log(f"  FPS: {1000/avg:.1f}")
        log(f"  Найдено: {found}/{len(images_data)} ({100*found/len(images_data):.1f}%)")
        log(f"  Всего детекций: {total_dets}")
    except Exception as e:
        log(f"❌ Ошибка: {e}")
    
    # === ТЕСТ 3: ONNX Runtime ===
    log("\n" + "="*70)
    log("ТЕСТ 3: Обученная модель (ONNX Runtime)")
    log("="*70)
    
    try:
        import onnxruntime as ort
        
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        session = ort.InferenceSession(onnx_path, sess_options=sess_options, 
                                        providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        output_names = [o.name for o in session.get_outputs()]
        
        log(f"✓ ONNX модель загружена")
        
        def preprocess_onnx(img):
            img_resized = cv2.resize(img, (320, 320))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_float = img_rgb.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_norm = (img_float - mean) / std
            return np.expand_dims(np.transpose(img_norm, (2, 0, 1)), axis=0).astype(np.float32)
        
        # Warmup
        for _ in range(3):
            session.run(output_names, {input_name: preprocess_onnx(images_data[0][1])})
        
        times = []
        for img_path, img in images_data:
            t0 = time.time()
            session.run(output_names, {input_name: preprocess_onnx(img)})
            times.append((time.time() - t0) * 1000)
        
        avg = np.mean(times)
        results['onnx'] = {
            'time': avg,
            'fps': 1000/avg
        }
        
        log(f"  Среднее время: {avg:.1f} ms")
        log(f"  FPS: {1000/avg:.1f}")
    except Exception as e:
        log(f"❌ Ошибка: {e}")
    
    # === ИТОГОВОЕ СРАВНЕНИЕ ===
    log("\n" + "="*70)
    log("ИТОГОВОЕ СРАВНЕНИЕ")
    log("="*70)
    log(f"{'Модель':<35} {'Время (ms)':<15} {'FPS':<10} {'Ускорение':<10}")
    log("-" * 70)
    
    base_time = results.get('onnx', {}).get('time', 1)
    
    if 'trained_ncnn' in results:
        r = results['trained_ncnn']
        speedup = base_time / r['time'] if 'onnx' in results else 1
        log(f"{'🚀 Обученная (NCNN)':<35} {r['time']:>6.1f} ms      {r['fps']:>5.1f}     {speedup:.1f}x")
    
    if 'model_zoo' in results:
        r = results['model_zoo']
        speedup = base_time / r['time'] if 'onnx' in results else 1
        log(f"{'📦 Model Zoo (NCNN)':<35} {r['time']:>6.1f} ms      {r['fps']:>5.1f}     {speedup:.1f}x")
    
    if 'onnx' in results:
        r = results['onnx']
        log(f"{'⏱️  Обученная (ONNX Runtime)':<35} {r['time']:>6.1f} ms      {r['fps']:>5.1f}     1.0x")
    
    log("\n" + "="*70)
    log("ВЫВОДЫ")
    log("="*70)
    
    if 'trained_ncnn' in results and 'onnx' in results:
        speedup = results['onnx']['time'] / results['trained_ncnn']['time']
        log(f"✅ Конвертация в NCNN ускорила модель в {speedup:.1f}x раз!")
        log(f"✅ Время инференса: {results['trained_ncnn']['time']:.1f} ms ({results['trained_ncnn']['fps']:.1f} FPS)")
        
        if 'model_zoo' in results:
            ratio = results['trained_ncnn']['time'] / results['model_zoo']['time']
            log(f"📊 Сравнение с Model Zoo: обученная модель в {ratio:.1f}x раз медленнее/быстрее")

if __name__ == "__main__":
    main()

