#!/usr/bin/env python3
"""
Детектор углов номерных знаков с использованием NCNN (быстрее чем ONNX на RPi)
"""
import os
import sys
import time
import argparse
import numpy as np
import cv2

try:
    import ncnn
except ImportError:
    print("Ошибка: NCNN не установлен!")
    print("Установите: pip install ncnn")
    sys.exit(1)

def order_corners(pts):
    """Sort corners: [top-left, top-right, bottom-right, bottom-left]."""
    pts = pts.reshape(4, 2)
    s = pts.sum(axis=1)
    diff = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(diff)]
    bl = pts[np.argmin(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32).flatten()

class LPCornerDetectorNCNN:
    def __init__(self, model_dir, num_threads=4):
        """
        Args:
            model_dir: Директория с файлами .param и .bin
            num_threads: Количество потоков (по умолчанию 4 для RPi 4)
        """
        # Поддержка разных форматов имен файлов
        param_path = None
        bin_path = None
        
        # Варианты имен файлов
        possible_names = [
            ("lp_regressor.ncnn.param", "lp_regressor.ncnn.bin"),
            ("lp_regressor.param", "lp_regressor.bin"),
        ]
        
        for param_name, bin_name in possible_names:
            p = os.path.join(model_dir, param_name)
            b = os.path.join(model_dir, bin_name)
            if os.path.exists(p) and os.path.exists(b):
                param_path = p
                bin_path = b
                break
        
        # Если не найдено, ищем любой .param файл
        if param_path is None:
            for f in os.listdir(model_dir):
                if f.endswith('.param') or f.endswith('.ncnn.param'):
                    param_path = os.path.join(model_dir, f)
                    bin_path = os.path.join(model_dir, f.replace('.param', '.bin').replace('.ncnn.param', '.ncnn.bin'))
                    if os.path.exists(bin_path):
                        break
        
        if not os.path.exists(param_path) or not os.path.exists(bin_path):
            raise FileNotFoundError(f"NCNN модель не найдена в {model_dir}")
        
        # Инициализация NCNN с оптимизациями для ARM
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False  # CPU режим (быстрее на RPi 4)
        self.net.opt.num_threads = num_threads   # Количество потоков
        self.net.opt.use_fp16_packed = True      # Использовать FP16 packed
        self.net.opt.use_fp16_storage = True     # Использовать FP16 storage
        self.net.opt.use_fp16_arithmetic = True  # Использовать FP16 arithmetic
        self.net.opt.use_packing_layout = True   # Оптимизация layout
        self.net.opt.use_winograd_convolution = True # Оптимизация сверток
        self.net.opt.lightmode = True            # Легкий режим
        
        self.net.load_param(param_path)
        self.net.load_model(bin_path)
        
        print(f"NCNN модель загружена: {model_dir}")
        print(f"Потоков: {num_threads}")
        
        # Разрешение входа (из модели или по умолчанию)
        self.input_w = 512
        self.input_h = 288
        
        # Предвычисленные координатные каналы для CoordConv
        self.xx_channel = np.tile(np.linspace(0, 1, self.input_w), (self.input_h, 1)).astype(np.float32)
        self.yy_channel = np.tile(np.linspace(0, 1, self.input_h).reshape(-1, 1), (1, self.input_w)).astype(np.float32)
        
        # ImageNet stats
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(self, img_bgr):
        """Предобработка для NCNN"""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        
        # Нормализация
        img_norm = img_resized.astype(np.float32) * (1.0 / 255.0)
        img_norm = (img_norm - self.mean) / self.std
        
        # CHW
        img_chw = img_norm.transpose(2, 0, 1)
        
        # Добавляем координатные каналы (3 -> 5)
        input_data = np.concatenate([
            img_chw,
            self.xx_channel[np.newaxis, ...],
            self.yy_channel[np.newaxis, ...]
        ], axis=0)
        
        return input_data

    def detect(self, img_path, output_dir):
        fname = os.path.basename(img_path)
        print(f"\nProcessing {fname}...")
        
        # 1. Load Image
        t0 = time.time()
        img_raw = cv2.imread(img_path)
        if img_raw is None:
            print(f"Error reading {img_path}")
            return
        h_orig, w_orig = img_raw.shape[:2]
        t_load = time.time() - t0
        
        # 2. Preprocess
        t1 = time.time()
        input_data = self.preprocess(img_raw)
        t_prep = time.time() - t1
        
        # 3. Inference
        t2 = time.time()
        
        # Создаем NCNN Mat из данных [5, H, W]
        # NCNN Mat хранит данные в формате [C, H, W] (channels first)
        # input_data уже в формате [5, H, W] - идеально!
        input_data_f32 = input_data.astype(np.float32)
        
        # Создаем Mat и заполняем данными
        mat_in = ncnn.Mat(self.input_w, self.input_h, 5)
        
        # Заполняем данные через numpy array (более эффективно)
        mat_array = np.array(mat_in, copy=False)
        # NCNN Mat: [C, H, W] формат
        mat_array[:] = input_data_f32
        
        ex = self.net.create_extractor()
        ex.input("in0", mat_in)
        
        ret, mat_out = ex.extract("out0")
        if ret != 0:
            print(f"Ошибка инференса: {ret}")
            return
        
        # Конвертируем выход в numpy
        pred_coords = np.array(mat_out).reshape(4, 2)
        t_infer = time.time() - t2
        
        # 4. Post-process
        # Map from [-1, 1] to [0, 1]
        pred_norm = (pred_coords + 1.0) / 2.0
        
        # Scale to original size
        pts = pred_norm.copy()
        pts[:, 0] *= w_orig
        pts[:, 1] *= h_orig
        
        # Order corners
        pts_final = order_corners(pts)
        
        # 5. Visualize
        vis = img_raw.copy()
        cv2.polylines(vis, [pts_final.astype(np.int32).reshape(4, 2)], True, (0, 0, 255), 2)
        
        # Stats
        total_time = (time.time() - t0) * 1000
        net_time = t_infer * 1000
        
        cv2.putText(vis, f"Total: {total_time:.1f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, f"Net: {net_time:.1f}ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        out_path = os.path.join(output_dir, fname)
        cv2.imwrite(out_path, vis)
        
        print(f"  [Timing]")
        print(f"  Load:    {t_load*1000:.1f} ms")
        print(f"  Prep:    {t_prep*1000:.1f} ms")
        print(f"  Infer:   {t_infer*1000:.1f} ms")
        print(f"  TOTAL:   {total_time:.1f} ms")

def main():
    parser = argparse.ArgumentParser(description='License Plate Corner Detector (NCNN)')
    parser.add_argument('--input', default='input', help='Input directory')
    parser.add_argument('--output', default='output', help='Output directory')
    parser.add_argument('--model', default='model_ncnn', help='Directory with NCNN model (.param and .bin files)')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads (default: 4)')
    args = parser.parse_args()
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
    if not os.path.exists(args.input):
        print(f"Input directory '{args.input}' not found. Creating it.")
        os.makedirs(args.input)
    
    detector = LPCornerDetectorNCNN(args.model, num_threads=args.threads)
    
    # Warmup
    print("Warming up...")
    dummy = np.zeros((detector.input_h, detector.input_w, 3), dtype=np.uint8)
    detector.preprocess(dummy)
    
    files = [f for f in os.listdir(args.input) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not files:
        print(f"No images found in {args.input}")
        return
        
    for f in files:
        detector.detect(os.path.join(args.input, f), args.output)

if __name__ == '__main__':
    main()

