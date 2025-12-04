#!/usr/bin/env python3
"""
Чистый тест инференса - только нейросеть, без декодирования
"""
import cv2
import numpy as np
import time
import ncnn
import onnxruntime as ort
from pathlib import Path
import random

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def main():
    log("="*70)
    log("ТЕСТ ЧИСТОГО ИНФЕРЕНСА (без декодирования)")
    log("="*70)
    
    # Загрузка изображений
    input_dir = "../input"
    images = list(Path(input_dir).glob('*.jpg'))
    random.seed(42)
    images = random.sample(images, min(20, len(images)))
    
    images_data = []
    for p in images:
        img = cv2.imread(str(p))
        if img is not None:
            images_data.append(img)
    
    log(f"Загружено {len(images_data)} изображений")
    
    # Предподготовка данных
    def prepare_ncnn(img):
        img_resized = cv2.resize(img, (320, 320))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        mat_in = ncnn.Mat.from_pixels(img_rgb, ncnn.Mat.PixelType.PIXEL_RGB, 320, 320)
        mat_in.substract_mean_normalize([123.675, 116.28, 103.53], [0.01712475, 0.0175, 0.01742919])
        return mat_in
    
    def prepare_onnx(img):
        img_resized = cv2.resize(img, (320, 320))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_norm = (img_float - mean) / std
        return np.expand_dims(np.transpose(img_norm, (2, 0, 1)), axis=0).astype(np.float32)
    
    # Предподготовленные данные
    ncnn_inputs = [prepare_ncnn(img) for img in images_data]
    onnx_inputs = [prepare_onnx(img) for img in images_data]
    
    log("\n" + "="*70)
    log("ТЕСТ 1: Обученная модель (NCNN оптимизированная) - только инференс")
    log("="*70)
    
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = 4
    net.opt.use_fp16_packed = True
    net.opt.use_fp16_storage = True
    net.opt.use_packing_layout = True
    net.load_param("../export/nanodet_320_opt.param")
    net.load_model("../export/nanodet_320_opt.bin")
    
    # Warmup
    for _ in range(10):
        ex = net.create_extractor()
        ex.input("in0", ncnn_inputs[0])
        for name in ["out0", "out1", "out2", "out3", "out4", "out5"]:
            ex.extract(name)
    
    # Тест
    times = []
    for mat_in in ncnn_inputs:
        t0 = time.time()
        ex = net.create_extractor()
        ex.input("in0", mat_in)
        for name in ["out0", "out1", "out2", "out3", "out4", "out5"]:
            ex.extract(name)
        times.append((time.time() - t0) * 1000)
    
    avg_ncnn = np.mean(times)
    log(f"  Среднее время: {avg_ncnn:.1f} ms")
    log(f"  FPS: {1000/avg_ncnn:.1f}")
    log(f"  Мин/Макс: {np.min(times):.1f} / {np.max(times):.1f} ms")
    
    log("\n" + "="*70)
    log("ТЕСТ 2: Model Zoo NanoDet (NCNN) - только инференс")
    log("="*70)
    
    from ncnn.model_zoo import get_model
    detector_zoo = get_model("nanodet", target_size=320, prob_threshold=0.25, num_threads=4)
    
    # Warmup
    for _ in range(10):
        detector_zoo(images_data[0])
    
    # Тест
    times = []
    for img in images_data:
        t0 = time.time()
        detector_zoo(img)
        times.append((time.time() - t0) * 1000)
    
    avg_zoo = np.mean(times)
    log(f"  Среднее время: {avg_zoo:.1f} ms")
    log(f"  FPS: {1000/avg_zoo:.1f}")
    log(f"  Мин/Макс: {np.min(times):.1f} / {np.max(times):.1f} ms")
    
    log("\n" + "="*70)
    log("ТЕСТ 3: Обученная модель (ONNX Runtime) - только инференс")
    log("="*70)
    
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(
        "../export/nanodet_320_simplified.onnx",
        sess_options=sess_options,
        providers=['CPUExecutionProvider']
    )
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    
    # Warmup
    for _ in range(5):
        session.run(output_names, {input_name: onnx_inputs[0]})
    
    # Тест
    times = []
    for inp in onnx_inputs:
        t0 = time.time()
        session.run(output_names, {input_name: inp})
        times.append((time.time() - t0) * 1000)
    
    avg_onnx = np.mean(times)
    log(f"  Среднее время: {avg_onnx:.1f} ms")
    log(f"  FPS: {1000/avg_onnx:.1f}")
    log(f"  Мин/Макс: {np.min(times):.1f} / {np.max(times):.1f} ms")
    
    # Итоги
    log("\n" + "="*70)
    log("ИТОГОВОЕ СРАВНЕНИЕ (чистый инференс)")
    log("="*70)
    log(f"{'Модель':<35} {'Время (ms)':<15} {'FPS':<10}")
    log("-" * 60)
    log(f"{'Обученная (NCNN)':<35} {avg_ncnn:>6.1f} ms      {1000/avg_ncnn:>5.1f}")
    log(f"{'Model Zoo (NCNN + декод)':<35} {avg_zoo:>6.1f} ms      {1000/avg_zoo:>5.1f}")
    log(f"{'Обученная (ONNX Runtime)':<35} {avg_onnx:>6.1f} ms      {1000/avg_onnx:>5.1f}")
    
    log(f"\n⚡ NCNN быстрее ONNX в {avg_onnx/avg_ncnn:.1f}x раз!")
    log(f"📊 Model Zoo включает декодирование, поэтому медленнее чистого NCNN")

if __name__ == "__main__":
    main()

