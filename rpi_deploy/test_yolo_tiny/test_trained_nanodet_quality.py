#!/usr/bin/env python3
"""
Тестирование обученной NanoDet модели на разных разрешениях
Сравнение скорости и качества обнаружения
"""
import cv2
import numpy as np
import time
import os
import sys
import random
import onnxruntime as ort
from pathlib import Path
from collections import defaultdict

def log(msg):
    """Вывод с временной меткой"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()

class TrainedNanoDetONNX:
    """Детектор на базе обученной NanoDet модели через ONNX Runtime"""
    
    def __init__(self, onnx_path, input_size=320, conf_threshold=0.35, num_threads=4):
        self.input_size = input_size  # Размер для ресайза входного изображения
        self.model_input_size = 320  # Модель обучена на 320x320
        self.conf_threshold = conf_threshold
        
        # Настройка сессии ONNX Runtime для ARM
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = num_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Провайдеры: CPU для Raspberry Pi
        providers = ['CPUExecutionProvider']
        
        # Загрузка модели
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=providers
        )
        
        # Получаем имена входов и выходов
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Получаем реальный размер входа модели
        input_shape = self.session.get_inputs()[0].shape
        if len(input_shape) >= 3:
            self.model_input_size = input_shape[-1]  # Последний размер
        
        log(f"✓ Модель загружена: {onnx_path}")
        log(f"  - Вход модели: {self.model_input_size}x{self.model_input_size}")
        log(f"  - Ресайз входного изображения: {input_size}x{input_size}")
        log(f"  - Выходы: {len(self.output_names)}")
        log(f"  - Потоков: {num_threads}")
        
        # Определяем strides на основе выходов (уникальные)
        stride_set = set()
        for out_name in sorted(self.output_names):
            if 'stride8' in out_name:
                stride_set.add(8)
            elif 'stride16' in out_name:
                stride_set.add(16)
            elif 'stride32' in out_name:
                stride_set.add(32)
        
        if stride_set:
            self.strides = sorted(list(stride_set))
        else:
            # Если не удалось определить, используем стандартные
            self.strides = [8, 16, 32]
        
        log(f"  - Strides: {self.strides}")
    
    def preprocess(self, img):
        """Предобработка изображения"""
        orig_h, orig_w = img.shape[:2]
        
        # Сначала ресайзим до нужного размера (для тестирования разных разрешений)
        # Затем ресайзим до размера модели (320x320)
        # Resize с сохранением пропорций (letterbox) до input_size
        scale1 = min(self.input_size / orig_w, self.input_size / orig_h)
        new_w1 = int(orig_w * scale1)
        new_h1 = int(orig_h * scale1)
        
        img_resized1 = cv2.resize(img, (new_w1, new_h1))
        
        # Создаем canvas с нулями для input_size
        canvas1 = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        
        # Размещаем изображение по центру
        y_offset1 = (self.input_size - new_h1) // 2
        x_offset1 = (self.input_size - new_w1) // 2
        canvas1[y_offset1:y_offset1+new_h1, x_offset1:x_offset1+new_w1] = img_resized1
        
        # Теперь ресайзим до размера модели (320x320)
        scale2 = self.model_input_size / self.input_size
        img_resized2 = cv2.resize(canvas1, (self.model_input_size, self.model_input_size))
        
        # Конвертация BGR -> RGB
        img_rgb = cv2.cvtColor(img_resized2, cv2.COLOR_BGR2RGB)
        
        # Нормализация ImageNet: (x - mean) / std
        img_float = img_rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_float - mean) / std
        
        # Преобразование в формат [1, 3, H, W]
        img_tensor = np.transpose(img_norm, (2, 0, 1))
        img_tensor = np.expand_dims(img_tensor, axis=0)
        
        # Общий scale учитывает оба ресайза
        total_scale = scale1 * scale2
        total_x_offset = x_offset1 * scale2
        total_y_offset = y_offset1 * scale2
        
        return img_tensor, (orig_h, orig_w), (total_scale, total_x_offset, total_y_offset)
    
    def decode_predictions(self, outputs, orig_shape, transform_params):
        """Декодирование предсказаний в bounding boxes"""
        detections = []
        orig_h, orig_w = orig_shape
        scale, x_offset, y_offset = transform_params
        
        # Разделяем выходы на cls и reg
        cls_outputs = []
        reg_outputs = []
        
        for out_name in sorted(self.output_names):
            if 'cls' in out_name.lower():
                cls_outputs.append(outputs[out_name])
            elif 'reg' in out_name.lower():
                reg_outputs.append(outputs[out_name])
        
        # Если выходы не разделены, пытаемся определить по форме
        if not cls_outputs or not reg_outputs:
            # Альтернативный подход: определяем по количеству каналов
            for out_name in sorted(self.output_names):
                output = outputs[out_name]
                if len(output.shape) == 4:
                    if output.shape[1] == 1:  # cls
                        cls_outputs.append(output)
                    elif output.shape[1] == 4:  # reg
                        reg_outputs.append(output)
        
        # Декодируем для каждого stride
        for stride, cls_pred, reg_pred in zip(self.strides, cls_outputs, reg_outputs):
            if cls_pred is None or reg_pred is None:
                continue
                
            # Форма: [1, C, H, W]
            _, _, h, w = cls_pred.shape
            
            # Применяем сигмоиду к классам
            scores = 1.0 / (1.0 + np.exp(-cls_pred))  # sigmoid
            
            # Проходим по всем ячейкам
            for i in range(h):
                for j in range(w):
                    score = scores[0, 0, i, j] if len(scores.shape) == 4 else scores[0, i, j]
                    
                    if score < self.conf_threshold:
                        continue
                    
                    # Декодирование bbox
                    # reg_pred: [1, 4, H, W] -> [l, t, r, b]
                    l = reg_pred[0, 0, i, j]
                    t = reg_pred[0, 1, i, j]
                    r = reg_pred[0, 2, i, j]
                    b = reg_pred[0, 3, i, j]
                    
                    # Центр ячейки в координатах модели
                    cx = (j + 0.5) * stride
                    cy = (i + 0.5) * stride
                    
                    # Координаты bbox в координатах модели (320x320)
                    x1_model = (cx - l)
                    y1_model = (cy - t)
                    x2_model = (cx + r)
                    y2_model = (cy + b)
                    
                    # Преобразуем из координат модели (320x320) в координаты input_size
                    # Модель работает на 320x320, но мы ресайзили вход до input_size
                    x1_model = (x1_model / self.model_input_size) * self.input_size
                    y1_model = (y1_model / self.model_input_size) * self.input_size
                    x2_model = (x2_model / self.model_input_size) * self.input_size
                    y2_model = (y2_model / self.model_input_size) * self.input_size
                    
                    # Учитываем letterbox трансформацию (из input_size в оригинал)
                    x1_model = (x1_model - x_offset) / scale
                    y1_model = (y1_model - y_offset) / scale
                    x2_model = (x2_model - x_offset) / scale
                    y2_model = (y2_model - y_offset) / scale
                    
                    # Ограничиваем координаты исходным изображением
                    x1 = max(0, min(orig_w, x1_model))
                    y1 = max(0, min(orig_h, y1_model))
                    x2 = max(0, min(orig_w, x2_model))
                    y2 = max(0, min(orig_h, y2_model))
                    
                    # Проверяем валидность bbox
                    if x2 > x1 and y2 > y1:
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'score': float(score),
                            'class': 0  # Vehicle
                        })
        
        return detections
    
    def nms(self, detections, iou_threshold=0.5):
        """Non-Maximum Suppression"""
        if len(detections) == 0:
            return []
        
        # Сортируем по score
        detections = sorted(detections, key=lambda x: x['score'], reverse=True)
        
        keep = []
        while len(detections) > 0:
            best = detections[0]
            keep.append(best)
            detections = detections[1:]
            
            # Удаляем пересекающиеся боксы
            filtered = []
            for det in detections:
                iou = self.compute_iou(best['bbox'], det['bbox'])
                if iou < iou_threshold:
                    filtered.append(det)
            detections = filtered
        
        return keep
    
    def compute_iou(self, box1, box2):
        """Вычисление IoU между двумя bbox"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Пересечение
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Площади
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # IoU
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0
    
    def detect(self, img):
        """Детекция объектов на изображении"""
        # Предобработка
        img_tensor, orig_shape, transform_params = self.preprocess(img)
        
        # Инференс
        outputs = self.session.run(self.output_names, {self.input_name: img_tensor})
        outputs_dict = {name: output for name, output in zip(self.output_names, outputs)}
        
        # Декодирование
        detections = self.decode_predictions(outputs_dict, orig_shape, transform_params)
        
        # NMS
        detections = self.nms(detections, iou_threshold=0.5)
        
        return detections

def test_model(detector, image, conf_threshold=0.35):
    """Тест модели на изображении"""
    t0 = time.time()
    detections = detector.detect(image)
    infer_time = (time.time() - t0) * 1000
    
    return detections, infer_time

def draw_detections(image, detections, color=(0, 255, 0)):
    """Отрисовка детекций"""
    result = image.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        score = det['score']
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        label = f"Vehicle {score:.2f}"
        cv2.putText(result, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return result

def main():
    log("="*70)
    log("Тестирование обученной NanoDet модели на разных разрешениях")
    log("="*70)
    
    # Параметры
    model_path = "../export/nanodet_320_simplified.onnx"
    if not os.path.exists(model_path):
        model_path = "../export/nanodet_320.onnx"
    
    if not os.path.exists(model_path):
        log(f"❌ Модель не найдена: {model_path}")
        log("Проверьте наличие файла nanodet_320.onnx или nanodet_320_simplified.onnx в папке export/")
        sys.exit(1)
    
    input_dir = "../input"
    output_dir = "output_trained_nanodet"
    num_images = 30  # Количество рандомных изображений для теста
    resolutions = [192, 256, 320]
    conf_threshold = 0.35
    
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
            detectors[res] = TrainedNanoDetONNX(
                model_path,
                input_size=res,
                conf_threshold=conf_threshold,
                num_threads=4
            )
        except Exception as e:
            log(f"❌ Ошибка загрузки детектора {res}x{res}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
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
                _ = detectors[res].detect(warmup_img)
        log("✓ Warmup завершен")
    
    # Статистика по разрешениям
    stats = defaultdict(lambda: {
        'times': [],
        'found_count': 0,
        'not_found_count': 0,
        'total_detections': 0,
        'avg_confidence': [],
        'detections_per_image': []
    })
    
    # Тестируем каждое изображение на всех разрешениях
    log("\n--- Тестирование ---")
    log(f"Модель: {model_path}")
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
            
            avg_conf = np.mean([d['score'] for d in detections]) if detections else 0
            
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
                stats[res]['avg_confidence'].extend([d['score'] for d in detections])
            
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
    
    log(f"\n✓ Результаты сохранены в: {output_dir}/")
    log(f"✓ Мозаики сравнения сохранены для каждого изображения")
    log("\n✓ Тест завершен!")

if __name__ == "__main__":
    main()

