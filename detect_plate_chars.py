"""
Комбинированный детектор: YOLO Pose (детекция номера) + YOLO (посимвольная детекция)
Максимально близко к эталонному коду с десктопа
"""
import os
import cv2
import numpy as np
import ncnn
import onnxruntime as ort
import yaml
import argparse
import time
from pathlib import Path


# Классы символов для маппинга (как в эталоне)
CHAR_CLASSES = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
    'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19,
    'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25,
    '0': 26, '1': 27, '2': 28, '3': 29, '4': 30, '5': 31, '6': 32, '7': 33, '8': 34, '9': 35
}
CLASS_NAMES = {v: k for k, v in CHAR_CLASSES.items()}


class NCNNPlateDetector:
    """Детектор номерных знаков на базе NCNN (YOLOv8-Pose)"""
    
    def __init__(self, model_dir, conf_threshold=0.25, iou_threshold=0.45):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        metadata_path = os.path.join(model_dir, "metadata.yaml")
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                meta = yaml.safe_load(f)
            imgsz = meta.get('imgsz', [640, 640])
            self.img_h = imgsz[0]
            self.img_w = imgsz[1]
        else:
            self.img_h = 640
            self.img_w = 640
        
        param_path = os.path.join(model_dir, "model.ncnn.param")
        bin_path = os.path.join(model_dir, "model.ncnn.bin")
        
        if not os.path.exists(param_path) or not os.path.exists(bin_path):
            raise FileNotFoundError(f"Не найдены файлы модели в {model_dir}")
        
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = 4
        self.net.load_param(param_path)
        self.net.load_model(bin_path)
    
    def preprocess(self, image):
        h, w = image.shape[:2]
        scale = min(self.img_h / h, self.img_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        padded.fill(114)
        pad_h = (self.img_h - new_h) // 2
        pad_w = (self.img_w - new_w) // 2
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        
        return padded, scale, (pad_w, pad_h)
    
    def detect(self, image):
        orig_h, orig_w = image.shape[:2]
        padded, scale, (pad_w, pad_h) = self.preprocess(image)
        
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        mat_in = ncnn.Mat.from_pixels(rgb, ncnn.Mat.PixelType.PIXEL_RGB, self.img_w, self.img_h)
        
        mean = [0, 0, 0]
        norm = [1/255.0, 1/255.0, 1/255.0]
        mat_in.substract_mean_normalize(mean, norm)
        
        ex = self.net.create_extractor()
        ex.input("in0", mat_in)
        ret, mat_out = ex.extract("out0")
        
        if ret != 0:
            return [], [], []
        
        output = np.array(mat_out)
        if len(output.shape) == 3:
            output = output[0]
        if output.shape[0] < output.shape[1]:
            output = output.T
        
        boxes = []
        scores = []
        keypoints = []
        
        for det in output:
            conf = det[4]
            if conf > 1.0:
                conf = 1.0 / (1.0 + np.exp(-conf))
            
            if conf < self.conf_threshold:
                continue
            
            cx, cy, w, h = det[0:4]
            x1 = (cx - w/2 - pad_w) / scale
            y1 = (cy - h/2 - pad_h) / scale
            x2 = (cx + w/2 - pad_w) / scale
            y2 = (cy + h/2 - pad_h) / scale
            
            x1 = np.clip(x1, 0, orig_w)
            y1 = np.clip(y1, 0, orig_h)
            x2 = np.clip(x2, 0, orig_w)
            y2 = np.clip(y2, 0, orig_h)
            
            boxes.append([x1, y1, x2, y2])
            scores.append(float(conf))
            
            if len(det) >= 17:
                kpts = []
                for i in range(4):
                    kx = (det[5 + i*3] - pad_w) / scale
                    ky = (det[5 + i*3 + 1] - pad_h) / scale
                    kv = det[5 + i*3 + 2]
                    kx = np.clip(kx, 0, orig_w)
                    ky = np.clip(ky, 0, orig_h)
                    kpts.append([kx, ky, kv])
                keypoints.append(kpts)
            else:
                keypoints.append([
                    [x1, y1, 1.0], [x2, y1, 1.0],
                    [x2, y2, 1.0], [x1, y2, 1.0]
                ])
        
        if len(boxes) > 0:
            boxes = np.array(boxes)
            scores = np.array(scores)
            indices = self.nms(boxes, scores)
            boxes = boxes[indices].tolist()
            scores = scores[indices].tolist()
            keypoints = [keypoints[i] for i in indices]
        
        return boxes, scores, keypoints
    
    def nms(self, boxes, scores):
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


class ONNXCharDetector:
    """Посимвольный детектор на базе ONNX (эквивалент ultralytics YOLO)"""
    
    def __init__(self, model_path, conf_threshold=0.25):
        self.conf_threshold = conf_threshold
        self.num_classes = 36
        
        # Загружаем модель ONNX
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        
        # Определяем размер входа из модели [batch, channels, height, width]
        self.img_h = int(input_shape[2]) if input_shape[2] else 640
        self.img_w = int(input_shape[3]) if input_shape[3] else 640
        
        print(f"OCR модель: {model_path}")
        print(f"  Вход: {self.img_w}x{self.img_h}")
    
    def detect(self, cropped_plate):
        """
        Детектирует символы на вырезанном номере.
        Возвращает список детекций в формате ultralytics:
        каждая детекция содержит xyxy координаты в пикселях исходного изображения (cropped_plate)
        """
        orig_h, orig_w = cropped_plate.shape[:2]
        
        # Предобработка (как в ultralytics)
        scale = min(self.img_h / orig_h, self.img_w / orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        
        resized = cv2.resize(cropped_plate, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        padded = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        padded.fill(114)
        pad_h = (self.img_h - new_h) // 2
        pad_w = (self.img_w - new_w) // 2
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        
        # Конвертируем в формат модели
        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        blob = blob.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)
        
        # Инференс
        outputs = self.session.run(None, {self.input_name: blob})
        output = outputs[0]
        
        # Обработка выхода YOLO
        if len(output.shape) == 3:
            output = output[0]
        if output.shape[0] < output.shape[1]:
            output = output.T
        
        # Формат: [x_center, y_center, width, height, class_0, ..., class_35] = 40 признаков
        detections = []
        
        for det in output:
            x_center, y_center, width, height = det[0:4]
            class_scores = det[4:4+self.num_classes]
            
            # Модель уже выводит вероятности (без softmax)
            class_id = class_scores.argmax()
            conf = class_scores[class_id]
            
            if conf < self.conf_threshold:
                continue
            
            # Преобразуем координаты из модели (640x640) в координаты cropped_plate
            # Это то, что ultralytics делает внутри и возвращает в box.xyxy[0]
            x1_model = x_center - width / 2
            y1_model = y_center - height / 2
            x2_model = x_center + width / 2
            y2_model = y_center + height / 2
            
            # Убираем padding и масштабируем
            x1 = (x1_model - pad_w) / scale
            y1 = (y1_model - pad_h) / scale
            x2 = (x2_model - pad_w) / scale
            y2 = (y2_model - pad_h) / scale
            
            # Clip к границам
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            
            # Проверяем размер
            if (x2 - x1) < 2 or (y2 - y1) < 2:
                continue
            if (x2 - x1) > orig_w * 0.5:
                continue
            
            char = CLASS_NAMES.get(class_id, '?')
            
            detections.append({
                'xyxy': [x1, y1, x2, y2],
                'class_id': class_id,
                'char': char,
                'conf': float(conf)
            })
        
        # NMS
        if len(detections) > 1:
            boxes = np.array([d['xyxy'] for d in detections])
            scores = np.array([d['conf'] for d in detections])
            keep = self.nms(boxes, scores, 0.5)
            detections = [detections[i] for i in keep]
        
        # Сортируем слева направо (как в эталоне)
        detections.sort(key=lambda d: d['xyxy'][0])
        
        return detections
    
    def nms(self, boxes, scores, iou_threshold=0.5):
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
            
            order = rest[iou <= iou_threshold]
        
        return keep


def crop_plate_from_keypoints(image, keypoints, keypoints_conf=None, padding=10):
    """Вырезает номер из изображения используя keypoints (как в эталоне)"""
    img_height, img_width = image.shape[:2]
    
    kp_array = np.array(keypoints)
    
    if len(kp_array.shape) == 2 and kp_array.shape[1] >= 2:
        x_coords = kp_array[:, 0]
        y_coords = kp_array[:, 1]
        
        if keypoints_conf is not None:
            conf_array = np.array(keypoints_conf)
            if len(conf_array) == len(x_coords):
                valid_mask = conf_array > 0.5
                if np.any(valid_mask):
                    x_coords = x_coords[valid_mask]
                    y_coords = y_coords[valid_mask]
    else:
        return None, None
    
    x_min = max(0, int(np.min(x_coords)) - padding)
    y_min = max(0, int(np.min(y_coords)) - padding)
    x_max = min(img_width, int(np.max(x_coords)) + padding)
    y_max = min(img_height, int(np.max(y_coords)) + padding)
    
    cropped = image[y_min:y_max, x_min:x_max]
    
    if cropped.size == 0:
        return None, None
    
    return cropped, (x_min, y_min, x_max, y_max)


def generate_color_palette(n):
    """Генерирует n различных цветов (как в эталоне)"""
    colors = []
    for i in range(n):
        hue = int(180 * i / n)
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in color))
    return colors


def process_images(input_dir, output_dir, plate_model_dir, char_model_path, 
                  plate_conf=0.25, char_conf=0.25):
    """Обработка изображений (как в эталоне)"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("Загрузка моделей...")
    plate_detector = NCNNPlateDetector(plate_model_dir, conf_threshold=plate_conf)
    char_detector = ONNXCharDetector(char_model_path, conf_threshold=char_conf)
    print("Готово!\n")
    
    # Получаем список изображений
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))
        image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"Не найдено изображений в: {input_dir}")
        return
    
    print(f"Найдено изображений: {len(image_files)}\n")
    
    stats = {'ok': 0, 'fail': 0}
    total_time = 0
    plate_time_total = 0
    char_time_total = 0
    other_time_total = 0
    
    for img_idx, img_path in enumerate(sorted(image_files)):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            h_img, w_img = img.shape[:2]
            start = time.time()
            
            # Детекция номеров
            plate_start = time.time()
            boxes, scores, keypoints = plate_detector.detect(img)
            plate_time = time.time() - plate_start
            plate_time_total += plate_time
            
            if len(boxes) == 0:
                stats['fail'] += 1
                print(f"❌ {img_path.name}: номера не найдены")
                continue
            
            # Берём лучший номер
            best_idx = np.argmax(scores)
            box = boxes[best_idx]
            score = scores[best_idx]
            kpts = keypoints[best_idx]
            
            # Вырезаем номер по keypoints (как в эталоне)
            kp_xy = np.array([[kpt[0], kpt[1]] for kpt in kpts])
            kp_conf = np.array([kpt[2] for kpt in kpts]) if len(kpts[0]) > 2 else None
            
            cropped_plate, plate_coords = crop_plate_from_keypoints(img, kp_xy, kp_conf)
            
            if cropped_plate is None:
                stats['fail'] += 1
                print(f"❌ {img_path.name}: не удалось вырезать номер")
                continue
            
            plate_x_min, plate_y_min, plate_x_max, plate_y_max = plate_coords
            
            # Распознавание символов на вырезанном номере
            char_start = time.time()
            char_detections = char_detector.detect(cropped_plate)
            char_time = time.time() - char_start
            char_time_total += char_time
            
            proc_time = time.time() - start
            other_time = proc_time - plate_time - char_time
            other_time_total += other_time
            total_time += proc_time
            
            # Собираем текст
            text = ''.join([d['char'] for d in char_detections])
            avg_conf = np.mean([d['conf'] for d in char_detections]) if char_detections else 0
            
            # Создаём визуализацию
            debug = img.copy()
            
            # Рисуем полигон номера по keypoints (как в эталоне)
            pts = np.array([[int(kpt[0]), int(kpt[1])] for kpt in kpts], dtype=np.int32)
            cv2.polylines(debug, [pts], True, (0, 255, 0), 2)
            
            # Рисуем ключевые точки с номерами
            for j, kpt in enumerate(kpts):
                x, y = int(kpt[0]), int(kpt[1])
                cv2.circle(debug, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(debug, str(j), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Рисуем символы с цветными рамками (как в эталоне)
            if char_detections:
                colors = generate_color_palette(len(char_detections))
                legend_items = []
                
                for i, det in enumerate(char_detections):
                    # Координаты символа в координатах cropped_plate
                    x1, y1, x2, y2 = det['xyxy']
                    
                    # Преобразуем в координаты исходного изображения (как в эталоне)
                    abs_x1 = x1 + plate_x_min
                    abs_y1 = y1 + plate_y_min
                    abs_x2 = x2 + plate_x_min
                    abs_y2 = y2 + plate_y_min
                    
                    char_pts = np.array([
                        [abs_x1, abs_y1],
                        [abs_x2, abs_y1],
                        [abs_x2, abs_y2],
                        [abs_x1, abs_y2]
                    ], dtype=np.int32)
                    
                    color = colors[i]
                    cv2.polylines(debug, [char_pts], True, color, 1)
                    
                    legend_items.append((i, det['char'], color, det['conf']))
                
                # Легенда (как в эталоне)
                legend_x, legend_y = 10, 30
                legend_spacing = 20
                bg_height = len(legend_items) * legend_spacing + 10
                
                overlay = debug.copy()
                cv2.rectangle(overlay, (legend_x - 5, legend_y - 20), 
                             (legend_x + 150, legend_y + bg_height), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, debug, 0.3, 0, debug)
                
                cv2.putText(debug, "Chars:", (legend_x, legend_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                for idx, (i, char, color, char_conf) in enumerate(legend_items):
                    y_pos = legend_y + (idx + 1) * legend_spacing
                    cv2.rectangle(debug, (legend_x, y_pos - 10), 
                                 (legend_x + 15, y_pos + 5), color, -1)
                    text_line = f"{i}: '{char}' ({char_conf:.2f})"
                    cv2.putText(debug, text_line, (legend_x + 20, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Сохраняем
            output_path = Path(output_dir) / f"{img_path.stem}_chars{img_path.suffix}"
            cv2.imwrite(str(output_path), debug)
            
            stats['ok'] += 1
            print(f"✅ {img_path.name}: {text} ({avg_conf:.2f}) [{proc_time*1000:.0f}ms]")
            
        except Exception as e:
            stats['fail'] += 1
            print(f"❌ {img_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*50)
    print("СТАТИСТИКА")
    print(f"✅ Успешно: {stats['ok']}")
    print(f"❌ Ошибок: {stats['fail']}")
    if stats['ok'] > 0:
        print(f"\nВРЕМЯ (среднее на изображение):")
        print(f"  1. Детекция номера (YOLO Pose NCNN): {plate_time_total/stats['ok']*1000:.0f} мс")
        print(f"  2. Детекция символов (ONNX 640x640): {char_time_total/stats['ok']*1000:.0f} мс")
        print(f"  3. Прочее (вырезка, визуализация):  {other_time_total/stats['ok']*1000:.0f} мс")
        print(f"  ─────────────────────────────────────")
        print(f"  ИТОГО: {total_time/stats['ok']*1000:.0f} мс")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description='Детектирование номеров + OCR')
    parser.add_argument('--input', type=str, default='./input_images')
    parser.add_argument('--output', type=str, default='./output_chars')
    parser.add_argument('--plate-model', type=str, default='./best_ncnn_model')
    parser.add_argument('--char-model', type=str, default='./ocr/best.onnx')
    parser.add_argument('--plate-conf', type=float, default=0.25)
    parser.add_argument('--char-conf', type=float, default=0.25)
    
    args = parser.parse_args()
    
    process_images(args.input, args.output, args.plate_model, args.char_model, 
                  args.plate_conf, args.char_conf)


if __name__ == '__main__':
    main()
