#!/usr/bin/env python3
"""
Тест NanoDet 256x256 для детекции автомобилей с оценкой качества
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

def test_model(detector, image, conf_threshold=0.25):
    """Тест модели на изображении"""
    # NanoDet сам обрабатывает изображение
    t0 = time.time()
    objects = detector(image)
    infer_time = (time.time() - t0) * 1000
    
    # Фильтруем только транспортные средства
    # Detect_Object имеет атрибуты: label, prob, rect (Rect объект)
    detections = []
    for obj in objects:
        label = obj.label
        prob = obj.prob
        rect = obj.rect
        
        # Rect имеет атрибуты x, y, w, h
        x = rect.x
        y = rect.y
        w = rect.w
        h = rect.h
        
        if label in VEHICLE_CLASSES and prob > conf_threshold:
            detections.append({
                'box': [int(x), int(y), int(w), int(h)],
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
        
        # Рисуем рамку
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Текст
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.putText(result, label, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return result

def main():
    log("="*60)
    log("Тест NanoDet 256x256 для детекции автомобилей")
    log("="*60)
    
    # Загрузка NanoDet
    log("\n--- Загрузка NanoDet ---")
    try:
        # NanoDet через ncnn model zoo
        # target_size=256 для входа 256x256
        detector = get_model("nanodet", target_size=256, prob_threshold=0.25, num_threads=4)
        log("✓ NanoDet загружен (NCNN backend)")
        log(f"✓ Входное разрешение: 256x256")
        log(f"✓ Потоков: 4")
    except Exception as e:
        log(f"❌ Ошибка загрузки NanoDet: {e}")
        sys.exit(1)
    
    # Тестовые изображения
    input_dir = "../input"
    output_dir = "output_nanodet_256"
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
    
    # Статистика
    times = []
    found_count = 0
    not_found_count = 0
    
    # Warmup (используем первое реальное изображение)
    log("--- Warmup (3 запуска) ---")
    if image_files:
        warmup_img = cv2.imread(str(image_files[0]))
        if warmup_img is not None:
            for i in range(3):
                _ = detector(warmup_img)
            log("✓ Warmup завершен")
    
    # Тестируем каждое изображение
    for img_idx, img_path in enumerate(image_files, 1):
        log(f"\n--- Изображение {img_idx}/{len(image_files)}: {img_path.name} ---")
        
        image = cv2.imread(str(img_path))
        if image is None:
            log(f"⚠ Не удалось загрузить {img_path.name}")
            continue
        
        orig_h, orig_w = image.shape[:2]
        log(f"Оригинальный размер: {orig_w}x{orig_h}")
        
        # Тест
        detections, infer_time = test_model(detector, image, conf_threshold=0.25)
        
        found = len(detections) > 0
        status = "✓ НАЙДЕНО" if found else "✗ НЕ НАЙДЕНО"
        
        log(f"  Время: {infer_time:5.1f}ms - {len(detections)} авто - {status}")
        
        if detections:
            for det in detections:
                log(f"    - {det['class_name']}: {det['confidence']:.2f}")
        
        times.append(infer_time)
        if found:
            found_count += 1
            # Сохраняем результат
            result_img = draw_detections(image, detections)
            output_path = os.path.join(output_dir, f"{img_path.stem}_detected.jpg")
            cv2.imwrite(output_path, result_img)
        else:
            not_found_count += 1
    
    # Итоговая статистика
    log("\n" + "="*60)
    log("ИТОГОВАЯ СТАТИСТИКА")
    log("="*60)
    
    if times:
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        fps = 1000 / avg_time
        total = found_count + not_found_count
        success_rate = (found_count / total * 100) if total > 0 else 0
        
        log(f"Модель: NanoDet 256x256 (NCNN)")
        log(f"Изображений протестировано: {total}")
        log(f"  Среднее время: {avg_time:.1f} ms")
        log(f"  Минимум:       {min_time:.1f} ms")
        log(f"  Максимум:      {max_time:.1f} ms")
        log(f"  Теоретический FPS: {fps:.1f}")
        log(f"  Найдено авто:  {found_count}/{total} ({success_rate:.1f}%)")
        log(f"  Не найдено:    {not_found_count}/{total} ({100-success_rate:.1f}%)")
        
        log(f"\n✓ Результаты сохранены в: {output_dir}/")
        
        if avg_time < 100:
            log(f"✓ Скорость приемлема для реального времени!")
        else:
            log(f"⚠ Скорость недостаточна для реального времени (нужно <100ms)")
    else:
        log("❌ Нет данных для статистики")
    
    log("\n✓ Тест завершен!")

if __name__ == "__main__":
    main()

