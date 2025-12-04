#!/usr/bin/env python3
"""
Тест YOLOv8n 256x256 для детекции автомобилей с оценкой качества
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

def postprocess_yolov8(output, conf_threshold=0.25, img_size=256):
    """Постобработка выходов YOLOv8"""
    detections = []
    
    # YOLOv8 output shape: [1, 84, 1344] где 84 = 4 координаты + 80 классов
    # Формат: [batch, features, detections]
    # features = [x_center, y_center, width, height, class_0, class_1, ..., class_79]
    
    if len(output.shape) == 3:
        output = output[0]  # Убираем batch dimension: [84, 1344]
    
    # Транспонируем: [features, detections] -> [detections, features]
    output = output.T  # [1344, 84]
    
    boxes = []
    confidences = []
    class_ids_list = []
    
    for detection in output:
        # YOLOv8 формат: [x_center, y_center, width, height, class_0, class_1, ...]
        x_center, y_center, width, height = detection[:4]
        
        # Получаем класс с максимальной вероятностью
        class_scores = detection[4:]
        class_id = np.argmax(class_scores)
        class_conf = class_scores[class_id]
        
        # Финальная уверенность
        final_conf = float(class_conf)
        
        if final_conf > conf_threshold and class_id in VEHICLE_CLASSES:
            # Координаты в пикселях (не нормализованы)
            x = int(x_center - width/2)
            y = int(y_center - height/2)
            w = int(width)
            h = int(height)
            
            # Проверяем валидность координат
            if x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= img_size and y + h <= img_size:
                boxes.append([x, y, w, h])
                confidences.append(final_conf)
                class_ids_list.append(int(class_id))
    
    # NMS для фильтрации дубликатов
    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.45)
        
        if len(indices) > 0:
            for i in indices.flatten():
                detections.append({
                    'box': boxes[i],
                    'confidence': confidences[i],
                    'class_id': class_ids_list[i],
                    'class_name': VEHICLE_CLASSES[class_ids_list[i]]
                })
    
    return detections

def test_model(net, image, conf_threshold=0.25):
    """Тест модели на изображении"""
    h, w = image.shape[:2]
    
    # YOLOv8 ожидает 256x256
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (256, 256), swapRB=True, crop=False)
    
    # Инференс
    t0 = time.time()
    net.setInput(blob)
    outputs = net.forward()
    infer_time = (time.time() - t0) * 1000
    
    # Постобработка
    detections = postprocess_yolov8(outputs[0], conf_threshold, img_size=256)
    
    return detections, infer_time

def draw_detections(image, detections, scale_x, scale_y):
    """Отрисовка детекций"""
    result = image.copy()
    
    for det in detections:
        x, y, w, h = det['box']
        # Масштабируем обратно
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
    log("Тест YOLOv8n 256x256 для детекции автомобилей")
    log("="*60)
    
    # Проверка модели
    MODEL_PATH = "yolov8n.onnx"
    
    if not os.path.exists(MODEL_PATH):
        log(f"❌ Модель не найдена: {MODEL_PATH}")
        log("Запустите: ./download_yolov8n_256.sh")
        sys.exit(1)
    
    # Загрузка сети
    log("\n--- Загрузка сети ---")
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    log("✓ Модель загружена (CPU backend)")
    log(f"✓ Входное разрешение: 256x256")
    
    # Тестовые изображения
    input_dir = "../input"
    output_dir = "output_yolov8n_256"
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
        detections, infer_time = test_model(net, image, conf_threshold=0.25)
        
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
            scale_x = orig_w / 256
            scale_y = orig_h / 256
            result_img = draw_detections(image, detections, scale_x, scale_y)
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
        
        log(f"Модель: YOLOv8n 256x256")
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

