"""
Комбинированный детектор: YOLO Pose (быстро) + EasyOCR (только на вырезанных областях)
Использует NCNN для быстрого детектирования номерных знаков,
затем применяет OCR только к найденным областям

Примечание: PaddleOCR имеет проблемы совместимости с ARM (segmentation fault),
поэтому используется EasyOCR, который лучше работает на Raspberry Pi
"""
import os
import cv2
import numpy as np
import ncnn
import easyocr
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
            self.img_h = imgsz[0]
            self.img_w = imgsz[1]
        else:
            self.img_h = 640
            self.img_w = 640
        
        # Загружаем модель NCNN
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
        """Предобработка изображения"""
        h, w = image.shape[:2]
        scale = min(self.img_h / h, self.img_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        padded = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        padded.fill(114)
        pad_h = (self.img_h - new_h) // 2
        pad_w = (self.img_w - new_w) // 2
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        
        return padded, scale, (pad_w, pad_h)
    
    def detect(self, image):
        """Детектирование номерных знаков"""
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
            
            # Ключевые точки (4 угла номера)
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


def extract_plate_region(image, keypoints):
    """
    Вырезает область номерного знака по ключевым точкам
    Использует perspective transform для правильной обрезки
    """
    # Преобразуем keypoints в numpy array
    src_pts = np.array([[kpt[0], kpt[1]] for kpt in keypoints], dtype=np.float32)
    
    # Вычисляем размеры выходного изображения
    # Используем ширину и высоту bounding box
    width = int(np.linalg.norm(src_pts[1] - src_pts[0]))
    height = int(np.linalg.norm(src_pts[2] - src_pts[1]))
    
    # Если точки не в правильном порядке, используем bbox
    if width < 10 or height < 10:
        x_coords = [pt[0] for pt in src_pts]
        y_coords = [pt[1] for pt in src_pts]
        width = int(max(x_coords) - min(x_coords))
        height = int(max(y_coords) - min(y_coords))
    
    # Минимальные размеры
    width = max(width, 50)
    height = max(height, 20)
    
    # Целевые точки для perspective transform
    dst_pts = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ], dtype=np.float32)
    
    # Вычисляем матрицу преобразования
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Применяем преобразование
    warped = cv2.warpPerspective(image, M, (width, height))
    
    return warped


def process_images(input_dir, output_dir, model_dir, conf_threshold=0.25):
    """Обработка изображений: детекция + OCR"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Инициализируем детектор
    print("Загрузка детектора NCNN...")
    detector = NCNNPlateDetector(model_dir, conf_threshold=conf_threshold)
    
    # Инициализируем OCR (EasyOCR лучше работает на ARM)
    print("Инициализация EasyOCR...")
    # EasyOCR: gpu=False для CPU, lang=['en'] для английских номеров
    ocr_engine = easyocr.Reader(['en'], gpu=False, verbose=False)
    print("EasyOCR готов!\n")
    
    # Прогрев
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    for _ in range(2):
        detector.detect(dummy)
    print("Прогрев завершён\n")
    
    # Получаем список изображений
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))
        image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"Не найдено изображений в: {input_dir}")
        return
    
    print(f"Найдено изображений: {len(image_files)}\n")
    
    total_time = 0
    total_detections = 0
    total_texts = 0
    processed = 0
    
    for img_path in image_files:
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            start = time.time()
            
            # Шаг 1: Быстрая детекция номеров
            boxes, scores, keypoints = detector.detect(image)
            detect_time = time.time() - start
            
            ocr_time = 0
            recognized_texts = []
            
            # Шаг 2: OCR только на найденных областях
            if len(boxes) > 0:
                ocr_start = time.time()
                for i, (box, score, kpts) in enumerate(zip(boxes, scores, keypoints)):
                    try:
                        # Вырезаем область с номером
                        plate_region = extract_plate_region(image, kpts)
                        
                        # Применяем OCR к вырезанной области
                        # EasyOCR возвращает список: [(bbox, text, confidence), ...]
                        ocr_result = ocr_engine.readtext(plate_region)
                        
                        # Извлекаем текст из результата EasyOCR
                        text = ""
                        ocr_conf = 0.0
                        
                        if ocr_result and len(ocr_result) > 0:
                            # EasyOCR возвращает список детекций: [(bbox, text, confidence), ...]
                            # Объединяем все найденные тексты или берем самый уверенный
                            texts = []
                            confidences = []
                            for detection in ocr_result:
                                if len(detection) >= 3:
                                    detected_text = detection[1]
                                    confidence = detection[2]
                                    texts.append(detected_text)
                                    confidences.append(confidence)
                            
                            if texts:
                                # Объединяем все тексты в один номер
                                text = "".join(texts).strip()
                                # Средняя уверенность
                                ocr_conf = sum(confidences) / len(confidences) if confidences else 0.0
                        
                        if text:
                            recognized_texts.append({
                                'text': text,
                                'ocr_confidence': ocr_conf,
                                'detection_score': score,
                                'box': box,
                                'keypoints': kpts
                            })
                    except Exception as e:
                        print(f"  Ошибка OCR для детекции {i}: {e}")
                
                ocr_time = time.time() - ocr_start
            
            proc_time = time.time() - start
            total_time += proc_time
            total_detections += len(boxes)
            total_texts += len(recognized_texts)
            processed += 1
            
            # Рисуем результаты
            result_image = image.copy()
            
            for i, (box, score) in enumerate(zip(boxes, scores)):
                x1, y1, x2, y2 = [int(coord) for coord in box]
                
                # Рисуем bbox
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Рисуем keypoints если есть
                if i < len(keypoints):
                    kpts = keypoints[i]
                    for kpt in kpts:
                        x, y = int(kpt[0]), int(kpt[1])
                        cv2.circle(result_image, (x, y), 3, (0, 0, 255), -1)
                
                # Текст с номером если распознан
                text_info = f"{score:.2f}"
                if i < len(recognized_texts):
                    text_info = f"{recognized_texts[i]['text']} ({score:.2f})"
                
                # Фон для текста
                (tw, th), bl = cv2.getTextSize(text_info, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(result_image, (x1, y1 - th - bl - 5),
                             (x1 + tw, y1), (0, 255, 0), -1)
                cv2.putText(result_image, text_info, (x1, y1 - bl - 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Общая информация
            info_text = f"Det: {len(boxes)} | OCR: {len(recognized_texts)} | {proc_time*1000:.0f}ms"
            cv2.putText(result_image, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Сохраняем результат
            output_path = Path(output_dir) / f"{img_path.stem}_det_ocr{img_path.suffix}"
            cv2.imwrite(str(output_path), result_image)
            
            # Выводим информацию
            if recognized_texts:
                texts_str = " | ".join([f"{item['text']}({item['ocr_confidence']:.2f})" 
                                       for item in recognized_texts])
                print(f"[{processed}/{len(image_files)}] {img_path.name}: "
                      f"{len(boxes)} детекций, {len(recognized_texts)} текстов, "
                      f"{proc_time*1000:.0f} мс (детекция: {detect_time*1000:.0f} мс, OCR: {ocr_time*1000:.0f} мс)")
                print(f"  Номера: {texts_str}")
            else:
                print(f"[{processed}/{len(image_files)}] {img_path.name}: "
                      f"{len(boxes)} детекций, текст не распознан, {proc_time*1000:.0f} мс")
            
        except Exception as e:
            print(f"Ошибка {img_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
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
        print(f"Распознано текстов: {total_texts} (среднее: {total_texts/processed:.1f})")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(
        description='Детектирование номерных знаков + OCR (NCNN + EasyOCR)'
    )
    parser.add_argument(
        '--input', 
        type=str, 
        default='./input_images', 
        help='Входной каталог'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='./output_det_ocr', 
        help='Выходной каталог'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='./best_ncnn_model', 
        help='Каталог с NCNN моделью'
    )
    parser.add_argument(
        '--conf', 
        type=float, 
        default=0.25, 
        help='Порог уверенности детекции'
    )
    
    args = parser.parse_args()
    
    process_images(args.input, args.output, args.model, args.conf)


if __name__ == '__main__':
    main()

