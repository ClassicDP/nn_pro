#!/usr/bin/env python3
"""
Тест YOLOv3-tiny на минимальных разрешениях с оценкой качества детекции автомобилей
"""
import cv2
import numpy as np
import time
import os
import sys
from pathlib import Path

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

def postprocess_yolo(outputs, conf_threshold=0.25, nms_threshold=0.45, img_size=320):
    """Постобработка выходов YOLOv3-tiny"""
    boxes = []
    confidences = []
    class_ids = []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > conf_threshold and class_id in VEHICLE_CLASSES:
                center_x = int(detection[0] * img_size)
                center_y = int(detection[1] * img_size)
                w = int(detection[2] * img_size)
                h = int(detection[3] * img_size)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            results.append({
                'box': boxes[i],
                'confidence': confidences[i],
                'class_id': class_ids[i],
                'class_name': VEHICLE_CLASSES[class_ids[i]]
            })
    
    return results

def test_resolution(net, image, resolution, conf_threshold=0.15):
    """Тест на конкретном разрешении"""
    h, w = image.shape[:2]
    
    # Изменяем размер входного изображения
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (resolution, resolution), swapRB=True, crop=False)
    
    # Инференс
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    t0 = time.time()
    net.setInput(blob)
    outputs = net.forward(output_layers)
    infer_time = (time.time() - t0) * 1000
    
    # Постобработка (используем более низкий порог для маленьких разрешений)
    detections = postprocess_yolo(outputs, conf_threshold, img_size=resolution)
    
    return detections, infer_time

def draw_detections(image, detections, scale_x, scale_y):
    """Отрисовка детекций на оригинальном изображении"""
    result = image.copy()
    
    for det in detections:
        x, y, w, h = det['box']
        # Масштабируем обратно к оригинальному размеру
        x = int(x * scale_x)
        y = int(y * scale_y)
        w = int(w * scale_x)
        h = int(h * scale_y)
        
        # Рисуем рамку
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Текст
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.putText(result, label, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return result

def main():
    log("="*60)
    log("Тест YOLOv3-tiny на минимальных разрешениях")
    log("="*60)
    
    # Проверка модели
    CONFIG = "yolov3-tiny.cfg"
    WEIGHTS = "yolov3-tiny.weights"
    
    if not os.path.exists(CONFIG) or not os.path.exists(WEIGHTS):
        log("❌ Модель не найдена!")
        log("Запустите: ./download_model.sh")
        sys.exit(1)
    
    # Загрузка сети
    log("\n--- Загрузка сети ---")
    net = cv2.dnn.readNetFromDarknet(CONFIG, WEIGHTS)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    log("✓ Модель загружена (CPU backend)")
    
    # Тестовые разрешения (от минимального)
    resolutions = [160, 192, 224, 256, 320]
    
    # Путь к тестовым изображениям
    input_dir = "../input"
    output_dir = "output_minimal_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Получаем изображения
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))
    
    if not image_files:
        log(f"❌ Не найдено изображений в {input_dir}")
        sys.exit(1)
    
    # Берем первые 10 изображений для быстрого теста
    image_files = sorted(image_files)[:10]
    log(f"\n--- Тестирование на {len(image_files)} изображениях ---")
    
    # Результаты по разрешениям
    results_by_res = {r: {'times': [], 'found': 0, 'not_found': 0} for r in resolutions}
    
    # Тестируем каждое изображение на всех разрешениях
    for img_idx, img_path in enumerate(image_files, 1):
        log(f"\n--- Изображение {img_idx}/{len(image_files)}: {img_path.name} ---")
        
        image = cv2.imread(str(img_path))
        if image is None:
            log(f"⚠ Не удалось загрузить {img_path.name}")
            continue
        
        orig_h, orig_w = image.shape[:2]
        log(f"Оригинальный размер: {orig_w}x{orig_h}")
        
        for resolution in resolutions:
            # Для маленьких разрешений используем более низкий порог
            conf_thresh = 0.15 if resolution <= 192 else 0.25
            detections, infer_time = test_resolution(net, image, resolution, conf_threshold=conf_thresh)
            
            found = len(detections) > 0
            status = "✓ НАЙДЕНО" if found else "✗ НЕ НАЙДЕНО"
            
            log(f"  {resolution}x{resolution}: {infer_time:5.1f}ms - {len(detections)} авто - {status}")
            
            results_by_res[resolution]['times'].append(infer_time)
            if found:
                results_by_res[resolution]['found'] += 1
            else:
                results_by_res[resolution]['not_found'] += 1
            
            # Сохраняем результат для минимального разрешения с детекцией
            if resolution == min(resolutions) and found:
                scale_x = orig_w / resolution
                scale_y = orig_h / resolution
                result_img = draw_detections(image, detections, scale_x, scale_y)
                output_path = os.path.join(output_dir, f"{img_path.stem}_min{resolution}.jpg")
                cv2.imwrite(output_path, result_img)
    
    # Итоговая статистика
    log("\n" + "="*60)
    log("ИТОГОВАЯ СТАТИСТИКА")
    log("="*60)
    log(f"{'Разрешение':<12} {'Среднее (ms)':<15} {'Мин (ms)':<12} {'FPS':<8} {'Найдено':<10} {'Не найдено':<12} {'% Успех':<10}")
    log("-"*60)
    
    for resolution in resolutions:
        r = results_by_res[resolution]
        if r['times']:
            avg_time = np.mean(r['times'])
            min_time = np.min(r['times'])
            fps = 1000 / avg_time
            total = r['found'] + r['not_found']
            success_rate = (r['found'] / total * 100) if total > 0 else 0
            
            log(f"{resolution}x{resolution:<6} {avg_time:>6.1f}        {min_time:>6.1f}      {fps:>5.1f}   {r['found']:>4}/{total:<5} {r['not_found']:>4}/{total:<7} {success_rate:>5.1f}%")
    
    # Рекомендация
    log("\n" + "="*60)
    log("РЕКОМЕНДАЦИИ")
    log("="*60)
    
    # Находим минимальное разрешение с приемлемым качеством (>50% успех)
    best_res = None
    for resolution in resolutions:
        r = results_by_res[resolution]
        total = r['found'] + r['not_found']
        if total > 0:
            success_rate = r['found'] / total * 100
            avg_time = np.mean(r['times']) if r['times'] else float('inf')
            
            if success_rate >= 50 and (best_res is None or avg_time < results_by_res[best_res]['times'] and results_by_res[best_res]['times']):
                best_res = resolution
    
    if best_res:
        r = results_by_res[best_res]
        avg_time = np.mean(r['times'])
        success_rate = r['found'] / (r['found'] + r['not_found']) * 100
        log(f"✓ Рекомендуемое разрешение: {best_res}x{best_res}")
        log(f"  Скорость: {avg_time:.1f}ms ({1000/avg_time:.1f} FPS)")
        log(f"  Качество: {success_rate:.1f}% успешных детекций")
    else:
        log("⚠ Не найдено разрешение с приемлемым качеством (>50%)")
    
    log(f"\n✓ Результаты сохранены в: {output_dir}/")
    log("✓ Тест завершен!")

if __name__ == "__main__":
    main()

