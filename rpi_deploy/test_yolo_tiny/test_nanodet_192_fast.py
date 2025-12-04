#!/usr/bin/env python3
"""
Тест NanoDet-M 192x192 с максимальной оптимизацией загрузки
Уменьшает большие изображения при загрузке для ускорения
"""
import cv2
import numpy as np
import time
import os
import sys
from pathlib import Path

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

def load_image_fast(img_path, max_dimension=800):
    """
    Быстрая загрузка изображения с уменьшением больших изображений
    max_dimension: максимальный размер по большей стороне (None = без уменьшения)
    """
    t0 = time.time()
    
    # Загружаем изображение
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    load_time = (time.time() - t0) * 1000
    
    if img is None:
        return None, load_time, 0
    
    # Если изображение очень большое, уменьшаем его для ускорения предобработки
    # Модель сама сделает финальный ресайз до 192x192, но предварительное
    # уменьшение ускорит работу с большими изображениями
    resize_time = 0
    if max_dimension:
        h, w = img.shape[:2]
        max_size = max(h, w)
        
        if max_size > max_dimension:
            t_resize = time.time()
            scale = max_dimension / max_size
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # INTER_LINEAR быстрее INTER_AREA, но для уменьшения INTER_AREA лучше
            # Используем INTER_AREA для лучшего качества при уменьшении
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            resize_time = (time.time() - t_resize) * 1000
    
    return img, load_time, resize_time

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

def draw_detections(image, detections):
    """Отрисовка детекций"""
    result = image.copy()
    for det in detections:
        x, y, w, h = det['box']
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.putText(result, label, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return result

def main():
    log("="*60)
    log("Тест NanoDet-M 192x192 с оптимизированной загрузкой")
    log("="*60)
    
    # Загрузка NanoDet-M с разрешением 192x192
    log("\n--- Загрузка NanoDet-M ---")
    try:
        detector = get_model("nanodet", target_size=192, prob_threshold=0.25, num_threads=4)
        log("✓ NanoDet-M загружен (NCNN backend)")
        log(f"✓ Входное разрешение: 192x192")
        log(f"✓ Потоков: 4")
    except Exception as e:
        log(f"❌ Ошибка загрузки NanoDet: {e}")
        sys.exit(1)
    
    # Тестовые изображения
    input_dir = "../input"
    output_dir = "output_nanodet_192_fast"
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))
    
    if not image_files:
        log(f"❌ Не найдено изображений в {input_dir}")
        sys.exit(1)
    
    # Берем первые 10 изображений
    image_files = sorted(image_files)[:10]
    log(f"\n--- Тестирование на {len(image_files)} изображениях ---")
    log("Оптимизация: уменьшение больших изображений до 800px при загрузке")
    
    # Статистика по времени
    load_times = []
    resize_times = []
    infer_times = []
    total_times = []
    found_count = 0
    not_found_count = 0
    
    # Предзагрузка с оптимизацией
    log("\n--- Предзагрузка изображений (с оптимизацией) ---")
    preload_start = time.time()
    images_data = []
    for img_path in image_files:
        img, load_time, resize_time = load_image_fast(img_path, max_dimension=800)
        if img is not None:
            images_data.append((img_path, img, load_time, resize_time))
        else:
            log(f"⚠ Не удалось загрузить {img_path.name}")
    
    preload_total = (time.time() - preload_start) * 1000
    log(f"✓ Предзагружено {len(images_data)} изображений за {preload_total:.1f}ms")
    if images_data:
        avg_load = np.mean([x[2] for x in images_data])
        avg_resize = np.mean([x[3] for x in images_data])
        log(f"  Среднее время загрузки: {avg_load:.1f}ms")
        log(f"  Среднее время ресайза: {avg_resize:.1f}ms")
    
    # Warmup
    log("\n--- Warmup (3 запуска) ---")
    if images_data:
        warmup_img = images_data[0][1]
        for i in range(3):
            _ = detector(warmup_img)
        log("✓ Warmup завершен")
    
    # Тестируем каждое изображение
    for img_idx, (img_path, image, load_time, resize_time) in enumerate(images_data, 1):
        log(f"\n--- Изображение {img_idx}/{len(images_data)}: {img_path.name} ---")
        
        orig_h, orig_w = image.shape[:2]
        log(f"Размер после оптимизации: {orig_w}x{orig_h}")
        log(f"  Время загрузки: {load_time:.1f}ms")
        if resize_time > 0:
            log(f"  Время ресайза: {resize_time:.1f}ms")
        
        # Тест инференса
        t_total = time.time()
        detections, infer_time = test_model(detector, image, conf_threshold=0.25)
        total_time = (time.time() - t_total) * 1000
        
        found = len(detections) > 0
        status = "✓ НАЙДЕНО" if found else "✗ НЕ НАЙДЕНО"
        log(f"  Время инференса: {infer_time:5.1f}ms")
        log(f"  Всего времени:   {total_time:5.1f}ms - {len(detections)} авто - {status}")
        
        if detections:
            for det in detections:
                log(f"    - {det['class_name']}: {det['confidence']:.2f}")
        
        load_times.append(load_time)
        resize_times.append(resize_time)
        infer_times.append(infer_time)
        total_times.append(total_time)
        
        if found:
            found_count += 1
            result_img = draw_detections(image, detections)
            output_path = os.path.join(output_dir, f"{img_path.stem}_detected.jpg")
            cv2.imwrite(output_path, result_img)
        else:
            not_found_count += 1
    
    # Итоговая статистика
    log("\n" + "="*60)
    log("ИТОГОВАЯ СТАТИСТИКА")
    log("="*60)
    
    if infer_times:
        avg_load = np.mean(load_times)
        avg_resize = np.mean(resize_times)
        avg_infer = np.mean(infer_times)
        avg_total = np.mean(total_times)
        min_infer = np.min(infer_times)
        max_infer = np.max(infer_times)
        fps = 1000 / avg_infer
        total = found_count + not_found_count
        success_rate = (found_count / total * 100) if total > 0 else 0
        
        log(f"Модель: NanoDet-M 192x192 (NCNN)")
        log(f"Изображений протестировано: {total}")
        log(f"\nВРЕМЯ ЗАГРУЗКИ:")
        log(f"  Среднее: {avg_load:.1f} ms")
        log(f"\nВРЕМЯ РЕСАЙЗА (если применялся):")
        log(f"  Среднее: {avg_resize:.1f} ms")
        log(f"\nВРЕМЯ ИНФЕРЕНСА:")
        log(f"  Среднее: {avg_infer:.1f} ms")
        log(f"  Минимум: {min_infer:.1f} ms")
        log(f"  Максимум: {max_infer:.1f} ms")
        log(f"  Теоретический FPS: {fps:.1f}")
        log(f"\nОБЩЕЕ ВРЕМЯ (загрузка + ресайз + инференс):")
        log(f"  Среднее: {avg_total:.1f} ms")
        log(f"\nКАЧЕСТВО:")
        log(f"  Найдено авто:  {found_count}/{total} ({success_rate:.1f}%)")
        log(f"  Не найдено:    {not_found_count}/{total} ({100-success_rate:.1f}%)")
        
        log(f"\n✓ Результаты сохранены в: {output_dir}/")
        
        # Анализ узких мест
        log(f"\n--- АНАЛИЗ УЗКИХ МЕСТ ---")
        preprocess_time = avg_load + avg_resize
        preprocess_percent = (preprocess_time / avg_total) * 100 if avg_total > 0 else 0
        infer_percent = (avg_infer / avg_total) * 100 if avg_total > 0 else 0
        log(f"Предобработка (загрузка+ресайз): {preprocess_percent:.1f}% от общего времени")
        log(f"Инференс: {infer_percent:.1f}% от общего времени")
        
        if avg_infer < 100:
            log(f"✓ Скорость инференса приемлема для реального времени!")
        else:
            log(f"⚠ Скорость инференса недостаточна (нужно <100ms)")
        
        if preprocess_time > avg_infer * 0.5:
            log(f"⚠ Предобработка занимает много времени!")
            log(f"  Рекомендация: использовать предзагрузку в отдельном потоке")
        else:
            log(f"✓ Предобработка оптимизирована!")
    else:
        log("❌ Нет данных для статистики")
    
    log("\n✓ Тест завершен!")

if __name__ == "__main__":
    main()



