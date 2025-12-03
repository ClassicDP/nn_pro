#!/usr/bin/env python3
import os
import sys
import time
import argparse
import numpy as np
import cv2
import onnxruntime as ort

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

class LPCornerDetectorRpi:
    def __init__(self, model_path, num_threads=None):
        # Initialize ONNX Runtime with optimizations
        providers = ['CPUExecutionProvider']
        
        # Оптимизации для Raspberry Pi
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL  # Все оптимизации графа
        sess_options.enable_mem_pattern = True  # Оптимизация памяти
        sess_options.enable_cpu_mem_arena = True  # CPU memory arena
        
        # Настройка потоков
        if num_threads is not None:
            sess_options.intra_op_num_threads = num_threads
            sess_options.inter_op_num_threads = 1  # Один поток между операциями
        # Если num_threads=None, ONNX сам выберет оптимальное значение
        
        # Режим выполнения: PARALLEL обычно быстрее, но SEQUENTIAL может быть стабильнее
        # Для малых моделей SEQUENTIAL иногда лучше
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL if num_threads == 1 else ort.ExecutionMode.ORT_PARALLEL
        
        self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        
        # Выводим информацию о конфигурации
        actual_threads = sess_options.intra_op_num_threads if num_threads is not None else "auto"
        print(f"ONNX Runtime config: threads={actual_threads}, mode=PARALLEL, mem_optimization=True")
        
        # New Resolution from Training V4
        self.input_w = 512
        self.input_h = 288
        
        # Precompute coordinate grids for CoordConv
        self.xx_channel = np.tile(np.linspace(0, 1, self.input_w), (self.input_h, 1)).astype(np.float32)
        self.yy_channel = np.tile(np.linspace(0, 1, self.input_h).reshape(-1, 1), (1, self.input_w)).astype(np.float32)
        
        # ImageNet stats
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(self, img_bgr):
        """Optimized preprocessing: Resize -> Norm -> CoordConv."""
        # Быстрое преобразование цвета
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Используем INTER_LINEAR для быстрого ресайза
        img_resized = cv2.resize(img_rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        
        # Оптимизированная нормализация (векторизованная)
        img_norm = img_resized.astype(np.float32) * (1.0 / 255.0)  # Быстрее чем деление
        img_norm = (img_norm - self.mean) / self.std
        
        # CHW (channel-first)
        img_chw = img_norm.transpose(2, 0, 1)
        
        # Add Coord Channels (3 -> 5) - координатные каналы уже предвычислены
        input_data = np.concatenate([
            img_chw,
            self.xx_channel[np.newaxis, ...],
            self.yy_channel[np.newaxis, ...]
        ], axis=0)
        
        # Add Batch Dim
        return input_data[np.newaxis, ...]

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
        
        # 2. Inference
        t1 = time.time()
        input_tensor = self.preprocess(img_raw)
        t_prep = time.time() - t1
        
        t2 = time.time()
        # Output is [1, 4, 2] in [-1, 1] range (DSNT)
        pred_coords = self.session.run(None, {self.input_name: input_tensor})[0][0] 
        t_infer = time.time() - t2
        
        # 3. Post-process
        # Map from [-1, 1] to [0, 1]
        pred_norm = (pred_coords + 1.0) / 2.0
        
        # Scale to original size
        pts = pred_norm.reshape(4, 2)
        pts[:, 0] *= w_orig
        pts[:, 1] *= h_orig
        
        # Order corners
        pts_final = order_corners(pts)
        
        # 4. Visualize
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
    parser = argparse.ArgumentParser(description='License Plate Corner Detector for Raspberry Pi')
    parser.add_argument('--input', default='input', help='Input directory')
    parser.add_argument('--output', default='output', help='Output directory')
    parser.add_argument('--model', default='model/lp_regressor.onnx', 
                       help='Path to ONNX model or preset: "full" (lp_regressor.onnx), "quant" (lp_regressor_quant.onnx), or path')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads for ONNX (default: 4 for RPi 4)')
    args = parser.parse_args()
    
    # Поддержка пресетов моделей
    model_presets = {
        'full': 'model/lp_regressor.onnx',
        'quant': 'model/lp_regressor_quant.onnx',
        'quantized': 'model/lp_regressor_quant.onnx',
    }
    
    if args.model in model_presets:
        model_path = model_presets[args.model]
    else:
        model_path = args.model
    
    # Проверка существования модели
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        print(f"Available presets: {', '.join(model_presets.keys())}")
        print(f"Or provide full path to ONNX model")
        return
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
    if not os.path.exists(args.input):
        print(f"Input directory '{args.input}' not found. Creating it.")
        os.makedirs(args.input)
        
    print(f"Using model: {model_path}")
    detector = LPCornerDetectorRpi(model_path, num_threads=args.threads)
    
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
