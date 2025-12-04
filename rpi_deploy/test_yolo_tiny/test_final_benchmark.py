#!/usr/bin/env python3
"""
Финальный стабильный бенчмарк
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

def benchmark_ncnn(param_path, bin_path, images, warmup=20, runs=50):
    """Бенчмарк NCNN модели"""
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = 4
    net.opt.use_fp16_packed = True
    net.opt.use_fp16_storage = True
    net.opt.use_packing_layout = True
    net.load_param(param_path)
    net.load_model(bin_path)
    
    # Подготовка входа
    def prepare(img):
        img_resized = cv2.resize(img, (320, 320))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        mat_in = ncnn.Mat.from_pixels(img_rgb, ncnn.Mat.PixelType.PIXEL_RGB, 320, 320)
        mat_in.substract_mean_normalize([123.675, 116.28, 103.53], [0.01712475, 0.0175, 0.01742919])
        return mat_in
    
    inputs = [prepare(img) for img in images]
    
    # Warmup
    for i in range(warmup):
        ex = net.create_extractor()
        ex.input("in0", inputs[i % len(inputs)])
        for name in ["out0", "out1", "out2", "out3", "out4", "out5"]:
            ex.extract(name)
    
    # Benchmark
    times = []
    for i in range(runs):
        mat_in = inputs[i % len(inputs)]
        t0 = time.time()
        ex = net.create_extractor()
        ex.input("in0", mat_in)
        for name in ["out0", "out1", "out2", "out3", "out4", "out5"]:
            ex.extract(name)
        times.append((time.time() - t0) * 1000)
    
    return times

def benchmark_onnx(onnx_path, images, warmup=10, runs=50):
    """Бенчмарк ONNX модели"""
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(onnx_path, sess_options=sess_options,
                                    providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    
    def prepare(img):
        img_resized = cv2.resize(img, (320, 320))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_norm = (img_float - mean) / std
        return np.expand_dims(np.transpose(img_norm, (2, 0, 1)), axis=0).astype(np.float32)
    
    inputs = [prepare(img) for img in images]
    
    # Warmup
    for i in range(warmup):
        session.run(output_names, {input_name: inputs[i % len(inputs)]})
    
    # Benchmark
    times = []
    for i in range(runs):
        inp = inputs[i % len(inputs)]
        t0 = time.time()
        session.run(output_names, {input_name: inp})
        times.append((time.time() - t0) * 1000)
    
    return times

def benchmark_modelzoo(images, warmup=20, runs=50):
    """Бенчмарк Model Zoo"""
    from ncnn.model_zoo import get_model
    detector = get_model("nanodet", target_size=320, prob_threshold=0.25, num_threads=4)
    
    # Warmup
    for i in range(warmup):
        detector(images[i % len(images)])
    
    # Benchmark
    times = []
    for i in range(runs):
        img = images[i % len(images)]
        t0 = time.time()
        detector(img)
        times.append((time.time() - t0) * 1000)
    
    return times

def main():
    log("="*70)
    log("ФИНАЛЬНЫЙ СТАБИЛЬНЫЙ БЕНЧМАРК")
    log("="*70)
    
    # Загрузка изображений
    input_dir = "../input"
    all_images = list(Path(input_dir).glob('*.jpg'))
    random.seed(42)
    selected = random.sample(all_images, min(30, len(all_images)))
    
    images = []
    for p in selected:
        img = cv2.imread(str(p))
        if img is not None:
            images.append(img)
    
    log(f"Загружено {len(images)} изображений")
    log(f"Warmup: 20 итераций, Тест: 50 итераций")
    
    results = {}
    
    # NCNN оптимизированная
    log("\n--- NCNN (оптимизированная) ---")
    times = benchmark_ncnn("../export/nanodet_320_opt.param", 
                           "../export/nanodet_320_opt.bin", images)
    results['ncnn_opt'] = np.mean(times)
    log(f"  Среднее: {np.mean(times):.1f} ms, Медиана: {np.median(times):.1f} ms")
    log(f"  Мин/Макс: {np.min(times):.1f} / {np.max(times):.1f} ms")
    
    # NCNN неоптимизированная
    log("\n--- NCNN (pnnx) ---")
    times = benchmark_ncnn("../export/nanodet_320_simplified.ncnn.param",
                           "../export/nanodet_320_simplified.ncnn.bin", images)
    results['ncnn_pnnx'] = np.mean(times)
    log(f"  Среднее: {np.mean(times):.1f} ms, Медиана: {np.median(times):.1f} ms")
    
    # Model Zoo
    log("\n--- Model Zoo (NanoDet-M 320) ---")
    times = benchmark_modelzoo(images)
    results['model_zoo'] = np.mean(times)
    log(f"  Среднее: {np.mean(times):.1f} ms, Медиана: {np.median(times):.1f} ms")
    log(f"  (включает декодирование и NMS)")
    
    # ONNX Runtime
    log("\n--- ONNX Runtime ---")
    times = benchmark_onnx("../export/nanodet_320_simplified.onnx", images)
    results['onnx'] = np.mean(times)
    log(f"  Среднее: {np.mean(times):.1f} ms, Медиана: {np.median(times):.1f} ms")
    
    # NCNN FP16 (pnnx)
    # log("\n--- NCNN (FP16 pnnx) ---")
    # try:
    #     times = benchmark_ncnn("../export/nanodet_320_fp16.ncnn.param",
    #                            "../export/nanodet_320_fp16.ncnn.bin", images)
    #     results['ncnn_fp16'] = np.mean(times)
    #     log(f"  Среднее: {np.mean(times):.1f} ms, Медиана: {np.median(times):.1f} ms")
    # except Exception as e:
    #     log(f"  Ошибка NCNN FP16: {e}")

    # ONNX Runtime FP16
    log("\n--- ONNX Runtime (FP16) ---")
    try:
        times = benchmark_onnx("../export/nanodet_320_fp16.onnx", images)
        results['onnx_fp16'] = np.mean(times)
        log(f"  Среднее: {np.mean(times):.1f} ms, Медиана: {np.median(times):.1f} ms")
    except Exception as e:
        log(f"  Ошибка ONNX FP16: {e}")

    # Итоги
    log("\n" + "="*70)
    log("ИТОГОВОЕ СРАВНЕНИЕ")
    log("="*70)
    log(f"{'Модель':<40} {'Время (ms)':<15} {'FPS':<10}")
    log("-" * 65)
    
    for name, avg in sorted(results.items(), key=lambda x: x[1]):
        log(f"{name:<40} {avg:>6.1f} ms      {1000/avg:>5.1f}")
    
    fastest = min(results, key=results.get)
    slowest = max(results, key=results.get)
    speedup = results[slowest] / results[fastest]
    
    log(f"\n🏆 Самый быстрый: {fastest} ({results[fastest]:.1f} ms)")
    log(f"🐢 Самый медленный: {slowest} ({results[slowest]:.1f} ms)")
    log(f"⚡ Разница: {speedup:.1f}x")

if __name__ == "__main__":
    main()

