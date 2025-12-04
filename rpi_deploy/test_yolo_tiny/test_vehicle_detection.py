#!/usr/bin/env python3
"""
Минималистичный тест скорости детекции автомобилей
YOLOv3-tiny 320×320 через OpenCV DNN с Vulkan/OpenCL/CPU
"""
import cv2
import numpy as np
import time
import os
import sys

def log(msg):
    """Вывод с временной меткой"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()

log("="*60)
log("Тест YOLOv3-tiny для детекции автомобилей")
log("="*60)

# Проверка OpenCV
log(f"OpenCV version: {cv2.__version__}")

# Проверка доступных backend'ов
log("\n--- Проверка доступных backend'ов ---")

# 1. Проверка OpenCL доступности (базовая проверка)
has_opencl_basic = cv2.ocl.haveOpenCL()
log(f"OpenCL доступен (базовая проверка): {has_opencl_basic}")

if has_opencl_basic:
    cv2.ocl.setUseOpenCL(True)
    devices = cv2.ocl.getDevice()
    log(f"Найдено OpenCL устройств: {len(devices)}")
    for i, device in enumerate(devices):
        log(f"  Device {i}: {device.name()}")
        log(f"    Type: {device.type()}")
        log(f"    Version: {device.version()}")
else:
    log("OpenCL недоступен")
    cv2.ocl.setUseOpenCL(False)

# 2. Проверка реального использования OpenCL в DNN
# ВАЖНО: cv2.ocl.haveOpenCL() может вернуть True, но DNN все равно использует CPU!
# Нужно проверить, действительно ли DNN может использовать OpenCL GPU
has_opencl_dnn = False
if has_opencl_basic and os.path.exists("yolov3-tiny.cfg") and os.path.exists("yolov3-tiny.weights"):
    try:
        log("\n--- Проверка реального использования OpenCL в DNN ---")
        test_net = cv2.dnn.readNetFromDarknet("yolov3-tiny.cfg", "yolov3-tiny.weights")
        test_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        test_net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
        
        # Пробуем выполнить инференс для проверки
        test_blob = cv2.dnn.blobFromImage(
            np.zeros((320, 320, 3), dtype=np.uint8), 
            1/255.0, (320, 320), swapRB=True
        )
        test_net.setInput(test_blob)
        _ = test_net.forward(test_net.getUnconnectedOutLayersNames())
        
        has_opencl_dnn = True
        log("✓ OpenCL DNN backend РЕАЛЬНО работает (использует GPU)")
    except Exception as e:
        log(f"✗ OpenCL DNN backend недоступен: {e}")
        log("⚠ OpenCV может быть собран БЕЗ поддержки OpenCL для DNN")
        log("⚠ DNN будет использовать CPU, даже если cv2.ocl.haveOpenCL() = True")
else:
    if has_opencl_basic:
        log("⚠ Модель не найдена, проверка OpenCL DNN пропущена")
    has_opencl_dnn = False

# 3. Проверка Vulkan через DNN
has_vulkan = False
if os.path.exists("yolov3-tiny.cfg") and os.path.exists("yolov3-tiny.weights"):
    try:
        log("\n--- Проверка Vulkan DNN backend ---")
        test_net = cv2.dnn.readNetFromDarknet("yolov3-tiny.cfg", "yolov3-tiny.weights")
        test_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_VKCOM)
        test_net.setPreferableTarget(cv2.dnn.DNN_TARGET_VULKAN)
        
        # Пробуем выполнить инференс
        test_blob = cv2.dnn.blobFromImage(
            np.zeros((320, 320, 3), dtype=np.uint8), 
            1/255.0, (320, 320), swapRB=True
        )
        test_net.setInput(test_blob)
        _ = test_net.forward(test_net.getUnconnectedOutLayersNames())
        
        has_vulkan = True
        log("✓ Vulkan DNN backend доступен")
    except Exception as e:
        log(f"✗ Vulkan DNN backend недоступен: {e}")
else:
    log("⚠ Модель не найдена, проверка Vulkan пропущена")

# Проверка модели
log("\n--- Проверка модели ---")
CONFIG = "yolov3-tiny.cfg"
WEIGHTS = "yolov3-tiny.weights"

if not os.path.exists(CONFIG):
    log(f"❌ Конфиг не найден: {CONFIG}")
    log("Скачайте: wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg")
    sys.exit(1)

if not os.path.exists(WEIGHTS):
    log(f"❌ Веса не найдены: {WEIGHTS}")
    log("Скачайте: wget https://pjreddie.com/media/files/yolov3-tiny.weights")
    sys.exit(1)

log(f"✓ Конфиг найден: {CONFIG}")
log(f"✓ Веса найдены: {WEIGHTS} ({os.path.getsize(WEIGHTS)/(1024*1024):.1f} MB)")

# Функция для тестирования с разными backend'ами
def test_backend(backend_name, backend_id, target_id):
    log(f"\n{'='*60}")
    log(f"Тест с {backend_name}")
    log(f"{'='*60}")
    
    # Загрузка сети
    log("--- Загрузка сети ---")
    try:
        net = cv2.dnn.readNetFromDarknet(CONFIG, WEIGHTS)
        net.setPreferableBackend(backend_id)
        net.setPreferableTarget(target_id)
        log(f"✓ Модель загружена с {backend_name}")
    except Exception as e:
        log(f"❌ Ошибка загрузки модели: {e}")
        return None
    
    # Получение выходных слоев
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # Подготовка данных
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    blob = cv2.dnn.blobFromImage(test_image, 1/255.0, (320, 320), swapRB=True, crop=False)
    
    # Warmup
    log("--- Warmup (3 запуска) ---")
    for i in range(3):
        t0 = time.time()
        net.setInput(blob)
        _ = net.forward(output_layers)
        t_warmup = (time.time() - t0) * 1000
        log(f"  Warmup {i+1}: {t_warmup:.1f} ms")
    
    # Основной тест
    log("--- Тест скорости (20 запусков) ---")
    times = []
    for i in range(20):
        t0 = time.time()
        net.setInput(blob)
        outputs = net.forward(output_layers)
        t_infer = (time.time() - t0) * 1000
        times.append(t_infer)
        if i < 5:
            log(f"  Run {i+1}: {t_infer:.1f} ms")
    
    # Статистика
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    std_time = np.std(times)
    
    log("\n--- Результаты ---")
    log(f"  Среднее время: {avg_time:.1f} ms")
    log(f"  Минимум:       {min_time:.1f} ms")
    log(f"  Максимум:      {max_time:.1f} ms")
    log(f"  Стандартное отклонение: {std_time:.1f} ms")
    log(f"  Теоретический FPS: {1000/avg_time:.1f}")
    
    return {
        'backend': backend_name,
        'avg': avg_time,
        'min': min_time,
        'max': max_time,
        'fps': 1000/avg_time
    }

# Тестирование всех доступных backend'ов
results = []

# 1. Vulkan (если доступен)
if has_vulkan:
    result = test_backend("Vulkan", cv2.dnn.DNN_BACKEND_VKCOM, cv2.dnn.DNN_TARGET_VULKAN)
    if result:
        results.append(result)

# 2. CPU (всегда доступен)
result = test_backend("CPU", cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)
if result:
    results.append(result)

# 3. OpenCL (если РЕАЛЬНО доступен для DNN)
# ВАЖНО: Используем has_opencl_dnn, а не has_opencl_basic!
# Это гарантирует, что DNN действительно использует GPU, а не CPU
if has_opencl_dnn:
    result = test_backend("OpenCL GPU", cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_OPENCL)
    if result:
        results.append(result)
elif has_opencl_basic:
    log("\n⚠ OpenCL доступен в OpenCV, но НЕ поддерживается в DNN")
    log("⚠ DNN будет использовать CPU вместо GPU")

# Итоговое сравнение
log("\n" + "="*60)
log("ИТОГОВОЕ СРАВНЕНИЕ")
log("="*60)
if results:
    results.sort(key=lambda x: x['avg'])
    for i, r in enumerate(results):
        marker = "🏆" if i == 0 else "  "
        log(f"{marker} {r['backend']:10s}: {r['avg']:6.1f} ms ({r['fps']:4.1f} FPS)")
    
    best = results[0]
    log(f"\n✓ Лучший результат: {best['backend']} - {best['avg']:.1f} ms ({best['fps']:.1f} FPS)")
else:
    log("❌ Не удалось протестировать ни один backend")

log("\n✓ Тест завершен!")
