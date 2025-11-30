"""
Детектирование номерных знаков на изображениях
ONNX Runtime с оптимальными настройками (внутренний параллелизм на все ядра)
"""
import os
import cv2
import numpy as np
import onnxruntime as ort
import argparse
import time
from pathlib import Path


class YOLOPostProcessor:
    """Класс для пост-обработки результатов YOLO модели"""
    
    def __init__(self, conf_threshold=0.25, iou_threshold=0.45, img_h=320, img_w=320):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_h = img_h
        self.img_w = img_w
    
    def preprocess(self, image):
        """Предобработка изображения для YOLO модели"""
        h, w = image.shape[:2]
        
        scale = min(self.img_h / h, self.img_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        padded = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        padded.fill(114)
        pad_h = (self.img_h - new_h) // 2
        pad_w = (self.img_w - new_w) // 2
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        
        padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        padded = padded.astype(np.float32) / 255.0
        padded = np.transpose(padded, (2, 0, 1))
        padded = np.expand_dims(padded, axis=0)
        
        return padded, scale, (pad_w, pad_h)
    
    def postprocess(self, outputs, scale, pad, orig_shape):
        """Пост-обработка выходов модели YOLO"""
        pad_w, pad_h = pad
        orig_h, orig_w = orig_shape
        
        if isinstance(outputs, (list, tuple)):
            predictions = outputs[0]
        else:
            predictions = outputs
        
        if len(predictions.shape) == 3:
            predictions = predictions[0]
            if predictions.shape[0] < predictions.shape[1]:
                predictions = predictions.transpose(1, 0)
        
        num_detections, num_features = predictions.shape
        
        if num_features > 5:
            boxes_raw = predictions[:, :4]
            obj_conf = predictions[:, 4:5]
            class_conf = predictions[:, 5:]
            
            if obj_conf.max() > 1.0:
                obj_conf = 1.0 / (1.0 + np.exp(-np.clip(obj_conf, -500, 500)))
            if class_conf.max() > 1.0:
                exp_conf = np.exp(class_conf - class_conf.max(axis=1, keepdims=True))
                class_conf = exp_conf / exp_conf.sum(axis=1, keepdims=True)
            
            class_scores = obj_conf * class_conf.max(axis=1, keepdims=True)
            class_ids = class_conf.argmax(axis=1)
            scores = class_scores.flatten()
        elif num_features == 5:
            boxes_raw = predictions[:, :4]
            scores = predictions[:, 4]
            class_ids = np.zeros(len(scores), dtype=int)
        else:
            boxes_raw = predictions[:, :4]
            scores = predictions[:, -1] if num_features > 4 else np.ones(len(predictions))
            class_ids = np.zeros(len(scores), dtype=int)
        
        mask = scores > self.conf_threshold
        if not np.any(mask):
            return [], [], []
        
        boxes_raw = boxes_raw[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        
        x_center = boxes_raw[:, 0]
        y_center = boxes_raw[:, 1]
        width = boxes_raw[:, 2]
        height = boxes_raw[:, 3]
        
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        x1 = (x1 - pad_w) / scale
        y1 = (y1 - pad_h) / scale
        x2 = (x2 - pad_w) / scale
        y2 = (y2 - pad_h) / scale
        
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)
        
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        indices = self.nms(boxes, scores)
        
        return boxes[indices].tolist(), scores[indices].tolist(), class_ids[indices].tolist()
    
    def nms(self, boxes, scores):
        """Non-Maximum Suppression"""
        if len(boxes) == 0:
            return []
        
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        order = scores.argsort()[::-1]
        keep = []
        
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            
            if len(order) == 1:
                break
            
            rest = order[1:]
            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h
            
            union = areas[i] + areas[rest] - intersection
            iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union != 0)
            
            order = rest[iou <= self.iou_threshold]
        
        return np.array(keep)


class PlateDetector:
    """Детектор номерных знаков (ONNX)"""
    
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        self.model_path = model_path
        
        # Оптимальные настройки: ONNX сам использует все ядра CPU
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True
        # НЕ устанавливаем intra_op_num_threads - пусть ONNX сам выберет оптимальное значение
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        # Поддержка прямоугольного входа [batch, channels, height, width]
        self.img_h = input_shape[2] if len(input_shape) > 2 else 320
        self.img_w = input_shape[3] if len(input_shape) > 3 else self.img_h
        
        self.postprocessor = YOLOPostProcessor(
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            img_h=self.img_h,
            img_w=self.img_w
        )
        
        print(f"Модель: {model_path}")
        print(f"Вход: {self.img_w}x{self.img_h}")
    
    def detect(self, image):
        """Детектирование номерных знаков"""
        orig_shape = image.shape[:2]
        preprocessed, scale, pad = self.postprocessor.preprocess(image)
        outputs = self.session.run(None, {self.input_name: preprocessed})
        return self.postprocessor.postprocess(outputs, scale, pad, orig_shape)
    
    def draw_results(self, image, boxes, scores, class_ids):
        """Рисует результаты полигонами"""
        if len(boxes) == 0:
            return image
        
        polygons = []
        for box in boxes:
            x1, y1, x2, y2 = box
            pts = np.array([
                [int(x1), int(y1)],
                [int(x2), int(y1)],
                [int(x2), int(y2)],
                [int(x1), int(y2)]
            ], dtype=np.int32)
            polygons.append(pts)
        
        # Полупрозрачная заливка
        overlay = image.copy()
        cv2.fillPoly(overlay, polygons, (0, 255, 0))
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
        
        # Контуры и метки
        for pts, score in zip(polygons, scores):
            cv2.polylines(image, [pts], True, (0, 255, 0), 2)
            cv2.circle(image, tuple(pts[0]), 5, (0, 0, 255), -1)
            
            label = f"Plate: {score:.2f}"
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (pts[0][0], pts[0][1] - th - bl - 5),
                          (pts[0][0] + tw, pts[0][1]), (0, 255, 0), -1)
            cv2.putText(image, label, (pts[0][0], pts[0][1] - bl - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return image


def process_images(input_dir, output_dir, model_path, conf_threshold=0.25, iou_threshold=0.45):
    """Обрабатывает все изображения последовательно (оптимальный режим для ONNX)"""
    os.makedirs(output_dir, exist_ok=True)
    
    detector = PlateDetector(model_path, conf_threshold, iou_threshold)
    
    # Прогрев модели
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    for _ in range(3):
        detector.detect(dummy)
    print("Прогрев завершён")
    
    # Получаем список изображений
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))
        image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"Не найдено изображений в: {input_dir}")
        return
    
    print(f"Изображений: {len(image_files)}")
    print(f"Режим: последовательная обработка (ONNX использует все ядра CPU)")
    print()
    
    total_time = 0
    total_detections = 0
    processed = 0
    
    for img_path in image_files:
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Не удалось загрузить: {img_path.name}")
                continue
            
            start = time.time()
            boxes, scores, class_ids = detector.detect(image)
            proc_time = time.time() - start
            
            total_time += proc_time
            total_detections += len(boxes)
            processed += 1
            
            # Сохраняем результат
            if len(boxes) > 0:
                result = detector.draw_results(image.copy(), boxes, scores, class_ids)
                cv2.putText(result, f"Detections: {len(boxes)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                result = image
            
            output_path = Path(output_dir) / f"{img_path.stem}_detected{img_path.suffix}"
            cv2.imwrite(str(output_path), result)
            
            print(f"[{processed}/{len(image_files)}] {img_path.name}: "
                  f"{len(boxes)} детекций, {proc_time*1000:.0f} мс")
            
        except Exception as e:
            print(f"Ошибка {img_path.name}: {e}")
    
    # Статистика
    print("\n" + "="*50)
    print("СТАТИСТИКА")
    print("="*50)
    print(f"Обработано: {processed}/{len(image_files)}")
    print(f"Общее время: {total_time:.2f}с")
    if processed > 0:
        fps = processed / total_time
        ms_per_img = total_time / processed * 1000
        print(f"Время на изображение: {ms_per_img:.0f} мс")
        print(f"FPS: {fps:.1f}")
        print(f"Детекций: {total_detections} (среднее: {total_detections/processed:.1f})")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description='Детектирование номерных знаков')
    parser.add_argument('--input', type=str, required=True, help='Входной каталог')
    parser.add_argument('--output', type=str, required=True, help='Выходной каталог')
    parser.add_argument('--model', type=str, default='best.onnx', help='Путь к модели')
    parser.add_argument('--conf', type=float, default=0.25, help='Порог уверенности')
    parser.add_argument('--iou', type=float, default=0.45, help='Порог IoU')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input):
        print(f"Ошибка: каталог не существует: {args.input}")
        return
    
    if not os.path.isfile(args.model):
        print(f"Ошибка: модель не найдена: {args.model}")
        return
    
    process_images(args.input, args.output, args.model, args.conf, args.iou)


if __name__ == '__main__':
    main()
