"""
Детектирование номерных знаков с использованием NCNN модели (YOLOv8-Pose)
Экспортировано через: yolo export model=best.pt format=ncnn half=True
"""
import os
import cv2
import numpy as np
import ncnn
import argparse
import time
from pathlib import Path


class NCNNPlateDetector:
    """Детектор номерных знаков на базе NCNN (YOLOv8-Pose)"""
    
    def __init__(self, model_dir, conf_threshold=0.25, iou_threshold=0.45):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Читаем размер из metadata.yaml если есть
        metadata_path = os.path.join(model_dir, "metadata.yaml")
        if os.path.exists(metadata_path):
            import yaml
            with open(metadata_path) as f:
                meta = yaml.safe_load(f)
            imgsz = meta.get('imgsz', [640, 640])
            self.img_h = imgsz[0]  # height
            self.img_w = imgsz[1]  # width
        else:
            self.img_h = 640
            self.img_w = 640
        
        # Загружаем модель NCNN
        param_path = os.path.join(model_dir, "model.ncnn.param")
        bin_path = os.path.join(model_dir, "model.ncnn.bin")
        
        if not os.path.exists(param_path) or not os.path.exists(bin_path):
            raise FileNotFoundError(f"Не найдены файлы модели в {model_dir}")
        
        self.net = ncnn.Net()
        # Оптимизации
        self.net.opt.use_vulkan_compute = False  # CPU режим
        self.net.opt.num_threads = 4  # 4 потока для Raspberry Pi 4
        
        self.net.load_param(param_path)
        self.net.load_model(bin_path)
        
        print(f"NCNN модель загружена: {model_dir}")
        print(f"Вход: {self.img_w}x{self.img_h}")
        print(f"Потоков: {self.net.opt.num_threads}")
    
    def preprocess(self, image):
        """Предобработка изображения"""
        h, w = image.shape[:2]
        
        # Масштабируем с сохранением пропорций
        scale = min(self.img_h / h, self.img_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Padding до img_h x img_w
        padded = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        padded.fill(114)
        pad_h = (self.img_h - new_h) // 2
        pad_w = (self.img_w - new_w) // 2
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        
        return padded, scale, (pad_w, pad_h)
    
    def detect(self, image):
        """Детектирование номерных знаков"""
        orig_h, orig_w = image.shape[:2]
        
        # Предобработка
        padded, scale, (pad_w, pad_h) = self.preprocess(image)
        
        # Конвертируем в RGB и создаем NCNN Mat
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        
        # Создаем Mat из изображения (width, height)
        mat_in = ncnn.Mat.from_pixels(rgb, ncnn.Mat.PixelType.PIXEL_RGB, self.img_w, self.img_h)
        
        # Нормализация: [0, 255] -> [0, 1]
        mean = [0, 0, 0]
        norm = [1/255.0, 1/255.0, 1/255.0]
        mat_in.substract_mean_normalize(mean, norm)
        
        # Инференс
        ex = self.net.create_extractor()
        ex.input("in0", mat_in)
        
        ret, mat_out = ex.extract("out0")
        if ret != 0:
            print(f"Ошибка извлечения: {ret}")
            return [], [], []
        
        # Конвертируем в numpy
        output = np.array(mat_out)
        
        # Парсим результаты YOLOv8-Pose
        # Формат: [batch, features, detections] или [features, detections]
        if len(output.shape) == 3:
            output = output[0]
        
        # Транспонируем если нужно: [features, detections] -> [detections, features]
        if output.shape[0] < output.shape[1]:
            output = output.T
        
        # YOLOv8-Pose формат: [x, y, w, h, conf, kpt1_x, kpt1_y, kpt1_vis, kpt2_x, ...]
        # 4 bbox + 1 conf + 4 keypoints * 3 = 17 features
        
        boxes = []
        scores = []
        keypoints = []
        
        for det in output:
            conf = det[4]
            
            # Применяем sigmoid если значения > 1
            if conf > 1.0:
                conf = 1.0 / (1.0 + np.exp(-conf))
            
            if conf < self.conf_threshold:
                continue
            
            # Bbox (center x, y, w, h)
            cx, cy, w, h = det[0:4]
            
            # Конвертируем в углы
            x1 = (cx - w/2 - pad_w) / scale
            y1 = (cy - h/2 - pad_h) / scale
            x2 = (cx + w/2 - pad_w) / scale
            y2 = (cy + h/2 - pad_h) / scale
            
            # Clip к границам изображения
            x1 = np.clip(x1, 0, orig_w)
            y1 = np.clip(y1, 0, orig_h)
            x2 = np.clip(x2, 0, orig_w)
            y2 = np.clip(y2, 0, orig_h)
            
            boxes.append([x1, y1, x2, y2])
            scores.append(float(conf))
            
            # Ключевые точки (4 угла номера)
            if len(det) >= 17:  # 4 + 1 + 12 (4 точки * 3)
                kpts = []
                for i in range(4):
                    kx = (det[5 + i*3] - pad_w) / scale
                    ky = (det[5 + i*3 + 1] - pad_h) / scale
                    kv = det[5 + i*3 + 2]  # visibility
                    
                    kx = np.clip(kx, 0, orig_w)
                    ky = np.clip(ky, 0, orig_h)
                    
                    kpts.append([kx, ky, kv])
                keypoints.append(kpts)
            else:
                # Нет keypoints, используем углы bbox
                keypoints.append([
                    [x1, y1, 1.0],
                    [x2, y1, 1.0],
                    [x2, y2, 1.0],
                    [x1, y2, 1.0]
                ])
        
        # NMS
        if len(boxes) > 0:
            boxes = np.array(boxes)
            scores = np.array(scores)
            indices = self.nms(boxes, scores)
            
            boxes = boxes[indices].tolist()
            scores = scores[indices].tolist()
            keypoints = [keypoints[i] for i in indices]
        
        return boxes, scores, keypoints
    
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
            
            intersection = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            union = areas[i] + areas[rest] - intersection
            iou = intersection / (union + 1e-6)
            
            order = rest[iou <= self.iou_threshold]
        
        return np.array(keep)
    
    def draw_results(self, image, boxes, scores, keypoints):
        """Рисует результаты с полигонами по ключевым точкам"""
        if len(boxes) == 0:
            return image
        
        for box, score, kpts in zip(boxes, scores, keypoints):
            # Рисуем полигон по ключевым точкам
            pts = np.array([[int(kpt[0]), int(kpt[1])] for kpt in kpts], dtype=np.int32)
            
            # Полупрозрачная заливка
            overlay = image.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
            
            # Контур
            cv2.polylines(image, [pts], True, (0, 255, 0), 2)
            
            # Ключевые точки
            for i, kpt in enumerate(kpts):
                x, y, v = int(kpt[0]), int(kpt[1]), kpt[2]
                if v > 0.5:  # Видимая точка
                    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
                    cv2.putText(image, str(i), (x+5, y-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Метка
            label = f"Plate: {score:.2f}"
            x1, y1 = int(box[0]), int(box[1])
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x1, y1 - th - bl - 5), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(image, label, (x1, y1 - bl - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return image


def process_images(input_dir, output_dir, model_dir, conf_threshold=0.25):
    """Обработка изображений"""
    os.makedirs(output_dir, exist_ok=True)
    
    detector = NCNNPlateDetector(model_dir, conf_threshold=conf_threshold)
    
    # Прогрев
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    for _ in range(3):
        detector.detect(dummy)
    print("Прогрев завершён\n")
    
    # Изображения
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))
        image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"Не найдено изображений в: {input_dir}")
        return
    
    print(f"Изображений: {len(image_files)}")
    
    total_time = 0
    total_detections = 0
    processed = 0
    
    for img_path in image_files:
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            start = time.time()
            boxes, scores, keypoints = detector.detect(image)
            proc_time = time.time() - start
            
            total_time += proc_time
            total_detections += len(boxes)
            processed += 1
            
            # Сохраняем результат
            if len(boxes) > 0:
                result = detector.draw_results(image.copy(), boxes, scores, keypoints)
                cv2.putText(result, f"Detections: {len(boxes)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                result = image
            
            output_path = Path(output_dir) / f"{img_path.stem}_ncnn{img_path.suffix}"
            cv2.imwrite(str(output_path), result)
            
            print(f"[{processed}/{len(image_files)}] {img_path.name}: "
                  f"{len(boxes)} детекций, {proc_time*1000:.0f} мс")
            
        except Exception as e:
            print(f"Ошибка {img_path.name}: {e}")
    
    # Статистика
    print("\n" + "="*50)
    print("СТАТИСТИКА NCNN")
    print("="*50)
    print(f"Обработано: {processed}/{len(image_files)}")
    print(f"Общее время: {total_time:.2f}с")
    if processed > 0:
        fps = processed / total_time
        ms_per_img = total_time / processed * 1000
        print(f"Время на изображение: {ms_per_img:.0f} мс")
        print(f"FPS: {fps:.1f}")
        print(f"Детекций: {total_detections}")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description='Детектирование номерных знаков (NCNN)')
    parser.add_argument('--input', type=str, default='./input_images', help='Входной каталог')
    parser.add_argument('--output', type=str, default='./output_ncnn', help='Выходной каталог')
    parser.add_argument('--model', type=str, default='./best_ncnn_model', help='Каталог с NCNN моделью')
    parser.add_argument('--conf', type=float, default=0.25, help='Порог уверенности')
    
    args = parser.parse_args()
    
    process_images(args.input, args.output, args.model, args.conf)


if __name__ == '__main__':
    main()

