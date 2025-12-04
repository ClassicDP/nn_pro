#!/usr/bin/env python3
"""
Сравнение качества обнаружения на всех разрешениях (192, 256, 320)
Рандомный выбор изображений, без предварительного ресайза
"""
import cv2
import numpy as np
import time
import os
import sys
import random
from pathlib import Path
from collections import defaultdict

# Импорт NanoDet из ncnn model zoo
try:
    from ncnn.model_zoo import get_model
except ImportError:
    print("❌ ncnn не установлен. Установите: pip install ncnn")
    sys.exit(1)

# COCO классы для транспортных средств
VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorbike', 
    5: 'bus',
    7: 'truck'
}

def log(msg):
    """Вывод с временной меткой"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()

def test_model(detector, image, conf_threshold=0.25):
    """Тест модели на изображении"""
    t0 = time.time()
    objects = detector(image)
    infer_time = (time.time() - t0) * 1000
    
    detections = []
    for obj in objects:
        label = obj.label
        prob = obj.prob
        rect = obj.rect
        
        if label in VEHICLE_CLASSES and prob > conf_threshold:
            detections.append({
                'box': [int(rect.x), int(rect.y), int(rect.w), int(rect.h)],
                'confidence': float(prob),
                'class_id': int(label),
                'class_name': VEHICLE_CLASSES[int(label)]
            })
    
    return detections, infer_time

def draw_detections(image, detections, color=(0, 255, 0)):
    """Отрисовка детекций"""
    result = image.copy()
    for det in detections:
        x, y, w, h = det['box']
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.putText(result, label, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return result

def main():
    log("="*70)
    log("Сравнение качества обнаружения на всех разрешениях")
    log("="*70)
    
    # Параметры
    input_dir = "../input"
    output_dir = "output_quality_comparison"
    num_images = 30  # Количество рандомных изображений для теста
    resolutions = [192, 256, 320]
    conf_threshold = 0.25
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Находим все изображения
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))
    
    if not image_files:
        log(f"❌ Не найдено изображений в {input_dir}")
        sys.exit(1)
    
    # Рандомный выбор изображений
    random.seed(42)  # Для воспроизводимости
    selected_images = random.sample(image_files, min(num_images, len(image_files)))
    log(f"\n--- Выбрано {len(selected_images)} изображений (рандомно) ---")
    
    # Загружаем детекторы для всех разрешений
    log("\n--- Загрузка детекторов ---")
    detectors = {}
    for res in resolutions:
        try:
            detectors[res] = get_model("nanodet", target_size=res, prob_threshold=conf_threshold, num_threads=4)
            log(f"✓ NanoDet {res}x{res} загружен")
        except Exception as e:
            log(f"❌ Ошибка загрузки NanoDet {res}x{res}: {e}")
            sys.exit(1)
    
    # Статистика по разрешениям
    stats = defaultdict(lambda: {
        'times': [],
        'found_count': 0,
        'not_found_count': 0,
        'total_detections': 0,
        'avg_confidence': [],
        'detections_per_image': []
    })
    
    # Предзагрузка изображений
    log("\n--- Предзагрузка изображений ---")
    images_data = []
    for img_path in selected_images:
        img = cv2.imread(str(img_path))
        if img is not None:
            images_data.append((img_path, img))
        else:
            log(f"⚠ Не удалось загрузить {img_path.name}")
    
    log(f"✓ Предзагружено {len(images_data)} изображений")
    
    # Warmup для каждого детектора
    log("\n--- Warmup (3 запуска для каждого разрешения) ---")
    if images_data:
        warmup_img = images_data[0][1]
        for res in resolutions:
            for i in range(3):
                _ = detectors[res](warmup_img)
        log("✓ Warmup завершен")
    
    # Тестируем каждое изображение на всех разрешениях
    log("\n--- Тестирование ---")
    log(f"Конфиденс порог: {conf_threshold}")
    log(f"Тестируем каждое изображение на разрешениях: {resolutions}")
    
    for img_idx, (img_path, image) in enumerate(images_data, 1):
        log(f"\n--- Изображение {img_idx}/{len(images_data)}: {img_path.name} ---")
        orig_h, orig_w = image.shape[:2]
        log(f"Оригинальный размер: {orig_w}x{orig_h}")
        
        results_per_res = {}
        
        # Тестируем на каждом разрешении
        for res in resolutions:
            detections, infer_time = test_model(detectors[res], image, conf_threshold)
            
            found = len(detections) > 0
            status = "✓" if found else "✗"
            
            avg_conf = np.mean([d['confidence'] for d in detections]) if detections else 0
            
            log(f"  {res}x{res}: {infer_time:5.1f}ms - {len(detections)} авто {status} (conf: {avg_conf:.2f})")
            
            results_per_res[res] = {
                'detections': detections,
                'infer_time': infer_time,
                'found': found,
                'avg_conf': avg_conf
            }
            
            # Обновляем статистику
            stats[res]['times'].append(infer_time)
            stats[res]['total_detections'] += len(detections)
            stats[res]['detections_per_image'].append(len(detections))
            if detections:
                stats[res]['avg_confidence'].extend([d['confidence'] for d in detections])
            
            if found:
                stats[res]['found_count'] += 1
            else:
                stats[res]['not_found_count'] += 1
        
        # Сохраняем результаты для сравнения (только если найдено хотя бы на одном разрешении)
        if any(results_per_res[res]['found'] for res in resolutions):
            # Создаем мозаику с результатами для каждого разрешения
            h, w = image.shape[:2]
            scale = 0.3  # Уменьшаем для мозаики
            small_h, small_w = int(h * scale), int(w * scale)
            
            mosaic_parts = []
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Зеленый, Красный, Синий
            
            for idx, res in enumerate(resolutions):
                result_img = draw_detections(image, results_per_res[res]['detections'], colors[idx])
                result_small = cv2.resize(result_img, (small_w, small_h))
                
                # Добавляем подпись с разрешением
                cv2.putText(result_small, f"{res}x{res} ({len(results_per_res[res]['detections'])} det)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[idx], 2)
                mosaic_parts.append(result_small)
            
            # Объединяем в одну строку
            mosaic = np.hstack(mosaic_parts)
            output_path = os.path.join(output_dir, f"{img_path.stem}_comparison.jpg")
            cv2.imwrite(output_path, mosaic)
    
    # Итоговая статистика
    log("\n" + "="*70)
    log("ИТОГОВАЯ СТАТИСТИКА ПО РАЗРЕШЕНИЯМ")
    log("="*70)
    
    for res in resolutions:
        s = stats[res]
        total = s['found_count'] + s['not_found_count']
        success_rate = (s['found_count'] / total * 100) if total > 0 else 0
        
        avg_time = np.mean(s['times']) if s['times'] else 0
        min_time = np.min(s['times']) if s['times'] else 0
        max_time = np.max(s['times']) if s['times'] else 0
        fps = 1000 / avg_time if avg_time > 0 else 0
        
        avg_detections = np.mean(s['detections_per_image']) if s['detections_per_image'] else 0
        avg_conf = np.mean(s['avg_confidence']) if s['avg_confidence'] else 0
        
        log(f"\n--- {res}x{res} ---")
        log(f"  Производительность:")
        log(f"    Среднее время: {avg_time:.1f} ms")
        log(f"    Минимум:      {min_time:.1f} ms")
        log(f"    Максимум:     {max_time:.1f} ms")
        log(f"    Теоретический FPS: {fps:.1f}")
        log(f"  Качество:")
        log(f"    Найдено авто: {s['found_count']}/{total} ({success_rate:.1f}%)")
        log(f"    Не найдено:   {s['not_found_count']}/{total} ({100-success_rate:.1f}%)")
        log(f"    Всего детекций: {s['total_detections']}")
        log(f"    Среднее детекций на изображение: {avg_detections:.1f}")
        log(f"    Средняя уверенность: {avg_conf:.2f}")
    
    # Сравнительная таблица
    log("\n" + "="*70)
    log("СРАВНИТЕЛЬНАЯ ТАБЛИЦА")
    log("="*70)
    log(f"{'Разрешение':<12} {'Время (ms)':<15} {'FPS':<8} {'Найдено %':<12} {'Детекций/изобр':<15} {'Уверенность':<12}")
    log("-" * 70)
    for res in resolutions:
        s = stats[res]
        total = s['found_count'] + s['not_found_count']
        success_rate = (s['found_count'] / total * 100) if total > 0 else 0
        avg_time = np.mean(s['times']) if s['times'] else 0
        fps = 1000 / avg_time if avg_time > 0 else 0
        avg_detections = np.mean(s['detections_per_image']) if s['detections_per_image'] else 0
        avg_conf = np.mean(s['avg_confidence']) if s['avg_confidence'] else 0
        
        log(f"{res}x{res:<8} {avg_time:>6.1f} ms      {fps:>5.1f}   {success_rate:>5.1f}%        {avg_detections:>5.1f}           {avg_conf:>5.2f}")
    
    # Рекомендации
    log("\n" + "="*70)
    log("РЕКОМЕНДАЦИИ")
    log("="*70)
    
    # Находим лучшее разрешение по балансу скорость/качество
    best_speed = min(resolutions, key=lambda r: np.mean(stats[r]['times']) if stats[r]['times'] else float('inf'))
    best_quality = max(resolutions, key=lambda r: stats[r]['found_count'])
    
    log(f"Самое быстрое: {best_speed}x{best_speed} ({np.mean(stats[best_speed]['times']):.1f}ms)")
    log(f"Лучшее качество: {best_quality}x{best_quality} ({stats[best_quality]['found_count']}/{len(images_data)} найдено)")
    
    # Рекомендация для каскадного подхода
    log(f"\nДля каскадного подхода:")
    log(f"  Этап 1 (детекция авто): {best_speed}x{best_speed} - максимальная скорость")
    log(f"  Этап 2 (детекция номера): {best_quality}x{best_quality} - лучшее качество на ROI")
    
    log(f"\n✓ Результаты сохранены в: {output_dir}/")
    log(f"✓ Мозаики сравнения сохранены для каждого изображения")
    log("\n✓ Тест завершен!")

if __name__ == "__main__":
    main()

