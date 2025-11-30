"""
Диагностика производительности ONNX Runtime
Проверяем: внутренний параллелизм ONNX, влияние потоков, оптимальная конфигурация
"""
import os
import cv2
import numpy as np
import onnxruntime as ort
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

# Фикс для multiprocessing на некоторых системах
multiprocessing.set_start_method('spawn', force=True)


def create_session(model_path, intra_threads=None, inter_threads=1):
    """Создает ONNX сессию с указанными настройками потоков"""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    if intra_threads is not None:
        sess_options.intra_op_num_threads = intra_threads
    sess_options.inter_op_num_threads = inter_threads
    
    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=['CPUExecutionProvider']
    )
    return session


def preprocess(image, img_size=320):
    """Быстрая предобработка"""
    h, w = image.shape[:2]
    scale = min(img_size / h, img_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    padded = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    padded.fill(114)
    pad_h = (img_size - new_h) // 2
    pad_w = (img_size - new_w) // 2
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    
    padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    padded = padded.astype(np.float32) / 255.0
    padded = np.transpose(padded, (2, 0, 1))
    padded = np.expand_dims(padded, axis=0)
    
    return padded


def run_inference(session, input_name, preprocessed):
    """Выполняет инференс"""
    return session.run(None, {input_name: preprocessed})


def benchmark_single_thread(model_path, images, num_runs=3):
    """Бенчмарк: 1 поток ONNX, последовательная обработка"""
    print("\n" + "="*60)
    print("ТЕСТ 1: Последовательная обработка, ONNX threads=1")
    print("="*60)
    
    session = create_session(model_path, intra_threads=1)
    input_name = session.get_inputs()[0].name
    
    # Прогрев
    preprocessed = preprocess(images[0])
    for _ in range(3):
        run_inference(session, input_name, preprocessed)
    
    times = []
    for _ in range(num_runs):
        start = time.time()
        for img in images:
            preprocessed = preprocess(img)
            run_inference(session, input_name, preprocessed)
        times.append(time.time() - start)
    
    avg_time = np.mean(times)
    per_image = avg_time / len(images)
    print(f"  Время на {len(images)} изображений: {avg_time:.2f}с")
    print(f"  Время на 1 изображение: {per_image*1000:.1f} мс")
    print(f"  FPS: {len(images)/avg_time:.2f}")
    return per_image


def benchmark_onnx_threads(model_path, images, onnx_threads, num_runs=3):
    """Бенчмарк: N потоков ONNX, последовательная обработка"""
    print(f"\n" + "="*60)
    print(f"ТЕСТ 2: Последовательная обработка, ONNX threads={onnx_threads}")
    print("="*60)
    
    session = create_session(model_path, intra_threads=onnx_threads)
    input_name = session.get_inputs()[0].name
    
    # Прогрев
    preprocessed = preprocess(images[0])
    for _ in range(3):
        run_inference(session, input_name, preprocessed)
    
    times = []
    for _ in range(num_runs):
        start = time.time()
        for img in images:
            preprocessed = preprocess(img)
            run_inference(session, input_name, preprocessed)
        times.append(time.time() - start)
    
    avg_time = np.mean(times)
    per_image = avg_time / len(images)
    print(f"  Время на {len(images)} изображений: {avg_time:.2f}с")
    print(f"  Время на 1 изображение: {per_image*1000:.1f} мс")
    print(f"  FPS: {len(images)/avg_time:.2f}")
    return per_image


def benchmark_default(model_path, images, num_runs=3):
    """Бенчмарк: дефолтные настройки ONNX (авто)"""
    print("\n" + "="*60)
    print("ТЕСТ 3: Последовательная обработка, ONNX threads=AUTO (дефолт)")
    print("="*60)
    
    session = create_session(model_path, intra_threads=None)
    input_name = session.get_inputs()[0].name
    
    # Прогрев
    preprocessed = preprocess(images[0])
    for _ in range(3):
        run_inference(session, input_name, preprocessed)
    
    times = []
    for _ in range(num_runs):
        start = time.time()
        for img in images:
            preprocessed = preprocess(img)
            run_inference(session, input_name, preprocessed)
        times.append(time.time() - start)
    
    avg_time = np.mean(times)
    per_image = avg_time / len(images)
    print(f"  Время на {len(images)} изображений: {avg_time:.2f}с")
    print(f"  Время на 1 изображение: {per_image*1000:.1f} мс")
    print(f"  FPS: {len(images)/avg_time:.2f}")
    return per_image


# Глобальные переменные для multiprocessing
_session = None
_input_name = None

def init_worker(model_path, intra_threads):
    """Инициализация воркера для ProcessPoolExecutor"""
    global _session, _input_name
    _session = create_session(model_path, intra_threads=intra_threads)
    _input_name = _session.get_inputs()[0].name

def process_image_worker(img_data):
    """Обработка изображения в отдельном процессе"""
    global _session, _input_name
    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    preprocessed = preprocess(img)
    start = time.time()
    run_inference(_session, _input_name, preprocessed)
    return time.time() - start


def benchmark_multiprocess(model_path, image_files, num_workers, onnx_threads_per_worker):
    """Бенчмарк: несколько процессов, каждый со своей ONNX сессией"""
    print(f"\n" + "="*60)
    print(f"ТЕСТ 4: {num_workers} процессов × {onnx_threads_per_worker} ONNX threads каждый")
    print("="*60)
    
    # Читаем изображения как байты для передачи между процессами
    image_data = []
    for f in image_files:
        with open(f, 'rb') as fp:
            image_data.append(fp.read())
    
    start = time.time()
    
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_worker,
        initargs=(model_path, onnx_threads_per_worker)
    ) as executor:
        results = list(executor.map(process_image_worker, image_data))
    
    total_time = time.time() - start
    per_image = total_time / len(image_files)
    
    print(f"  Время на {len(image_files)} изображений: {total_time:.2f}с")
    print(f"  Время на 1 изображение (wall-clock): {per_image*1000:.1f} мс")
    print(f"  Среднее время инференса: {np.mean(results)*1000:.1f} мс")
    print(f"  FPS: {len(image_files)/total_time:.2f}")
    return per_image


def main():
    model_path = "best.onnx"
    input_dir = "./input_images"
    
    # Получаем количество ядер
    num_cores = multiprocessing.cpu_count()
    print(f"\n{'='*60}")
    print(f"ДИАГНОСТИКА ПРОИЗВОДИТЕЛЬНОСТИ ONNX RUNTIME")
    print(f"{'='*60}")
    print(f"Процессор: {num_cores} ядер")
    print(f"Модель: {model_path}")
    
    # Загружаем изображения (берём первые 20 для быстрого теста)
    image_extensions = {'.jpg', '.jpeg', '.png'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))
        image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
    
    image_files = sorted(image_files)[:20]  # Берём 20 изображений
    print(f"Тестовых изображений: {len(image_files)}")
    
    # Загружаем изображения в память
    images = [cv2.imread(str(f)) for f in image_files]
    images = [img for img in images if img is not None]
    print(f"Загружено в память: {len(images)}")
    
    # Тесты
    results = {}
    
    # Тест 1: 1 поток ONNX
    results['onnx_1'] = benchmark_single_thread(model_path, images)
    
    # Тест 2: 4 потока ONNX  
    results['onnx_4'] = benchmark_onnx_threads(model_path, images, 4)
    
    # Тест 3: Авто (все ядра)
    results['onnx_auto'] = benchmark_default(model_path, images)
    
    # Тест 4: 2 процесса × 2 ONNX потока
    results['mp_2x2'] = benchmark_multiprocess(model_path, image_files, 2, 2)
    
    # Тест 5: 4 процесса × 1 ONNX поток
    results['mp_4x1'] = benchmark_multiprocess(model_path, image_files, 4, 1)
    
    # Итоги
    print("\n" + "="*60)
    print("СВОДКА РЕЗУЛЬТАТОВ (мс на изображение)")
    print("="*60)
    for name, time_per_img in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {name:20s}: {time_per_img*1000:6.1f} мс  ({1/time_per_img:.1f} FPS)")
    
    print("\n" + "="*60)
    print("ВЫВОДЫ")
    print("="*60)
    
    best = min(results.items(), key=lambda x: x[1])
    print(f"  Лучший вариант: {best[0]} ({best[1]*1000:.1f} мс)")
    
    if results['onnx_1'] < results['onnx_4']:
        print("  ⚠ ONNX с 1 потоком быстрее чем с 4 - модель слишком маленькая для параллелизма")
    if results['onnx_auto'] < results['onnx_4']:
        print("  ✓ Авто-настройка ONNX работает хорошо")
    
    print("="*60)


if __name__ == '__main__':
    main()





