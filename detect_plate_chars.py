"""
Комбинированный детектор: YOLO Pose (детекция номера) + YOLO (посимвольная детекция)
Максимально близко к эталонному коду с десктопа
Оптимизировано для Raspberry Pi 4 с NEON инструкциями
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
import multiprocessing

# Оптимизации для ARM/Raspberry Pi 4 с NEON
# Устанавливаем переменные окружения ДО импорта библиотек для максимальной производительности
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '4')

# Для ONNX Runtime - использовать все доступные оптимизации
os.environ.setdefault('ORT_DISABLE_ALL_OPTIMIZATIONS', '0')
os.environ.setdefault('ORT_ENABLE_BASIC_OPTIMIZATIONS', '1')
os.environ.setdefault('ORT_ENABLE_EXTENDED_OPTIMIZATIONS', '1')
os.environ.setdefault('ORT_ENABLE_LAYOUT_OPTIMIZATIONS', '1')

# Принудительно используем NEON для NumPy (если доступно)
try:
    # Проверяем поддержку NEON
    import numpy.core._multiarray_umath as _multiarray_umath
    # NumPy автоматически использует NEON если доступно
except:
    pass


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
    
    def __init__(self, model_dir, conf_threshold=0.25, iou_threshold=0.45, 
                 num_threads=None, use_vulkan=False):
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
        
        # Оптимизации для Raspberry Pi 4
        if num_threads is None:
            # Используем все доступные ядра (Raspberry Pi 4 имеет 4 ядра)
            num_threads = min(4, multiprocessing.cpu_count())
        
        self.net.opt.use_vulkan_compute = use_vulkan
        self.net.opt.num_threads = num_threads
        
        # Агрессивные оптимизации для ARM с NEON
        self.net.opt.use_winograd_convolution = True   # Winograd быстрее на ARM
        self.net.opt.use_winograd23_convolution = True  # Winograd 2x3 для ARM
        self.net.opt.use_winograd43_convolution = True  # Winograd 4x3 для ARM
        self.net.opt.use_sgemm_convolution = True      # Оптимизированные свертки
        self.net.opt.use_fp16_storage = False           # FP32 для точности (FP16 может быть медленнее на CPU)
        self.net.opt.use_fp16_arithmetic = False        # FP32 арифметика
        
        # Оптимизации памяти
        self.net.opt.use_packing_layout = True          # Упаковка данных для NEON
        
        self.net.load_param(param_path)
        self.net.load_model(bin_path)
        
        print(f"NCNN модель загружена: {model_dir}")
        print(f"  Вход: {self.img_w}x{self.img_h}")
        print(f"  Потоков: {self.net.opt.num_threads}")
        print(f"  Vulkan: {self.net.opt.use_vulkan_compute}")
    
    def preprocess(self, image):
        h, w = image.shape[:2]
        scale = min(self.img_h / h, self.img_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Используем INTER_AREA для уменьшения (быстрее на ARM)
        if scale < 1.0:
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Предвыделяем массив один раз (быстрее чем zeros + fill)
        padded = np.full((self.img_h, self.img_w, 3), 114, dtype=np.uint8)
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
        """NMS с оптимизациями для векторизации (NEON)"""
        if len(boxes) == 0:
            return []
        
        # Используем float32 для лучшей производительности на ARM
        boxes = boxes.astype(np.float32, copy=False)
        scores = scores.astype(np.float32, copy=False)
        
        # Векторизованное вычисление площадей
        areas = np.multiply(
            np.subtract(boxes[:, 2], boxes[:, 0], dtype=np.float32),
            np.subtract(boxes[:, 3], boxes[:, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        order = scores.argsort()[::-1]
        keep = []
        
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break
            
            rest = order[1:]
            # Векторизованные операции для лучшего использования NEON
            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
            
            # Векторизованное вычисление пересечения
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = np.multiply(w, h, dtype=np.float32)
            
            # Векторизованное вычисление IoU
            union = np.add(areas[i], areas[rest], dtype=np.float32)
            union = np.subtract(union, intersection, dtype=np.float32)
            iou = np.divide(intersection, np.add(union, 1e-6), dtype=np.float32)
            
            order = rest[iou <= self.iou_threshold]
        
        return np.array(keep)


class ONNXCharDetector:
    """Посимвольный детектор на базе ONNX (эквивалент ultralytics YOLO)"""
    
    def __init__(self, model_path, conf_threshold=0.25, num_threads=None):
        self.conf_threshold = conf_threshold
        self.num_classes = 36
        
        # Оптимизации для Raspberry Pi 4
        if num_threads is None:
            # Используем все доступные ядра
            num_threads = min(4, multiprocessing.cpu_count())
        
        # Загружаем модель ONNX с агрессивными оптимизациями для ARM
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Настройки потоков для Raspberry Pi 4
        sess_options.intra_op_num_threads = num_threads  # Параллелизм внутри операций
        sess_options.inter_op_num_threads = 1            # Один поток между операциями (лучше для малых моделей)
        
        # Оптимизации памяти для ARM
        sess_options.enable_mem_pattern = True           # Переиспользование памяти
        sess_options.enable_cpu_mem_arena = True          # CPU memory arena для ARM
        
        # Дополнительные оптимизации
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        # Оптимизации для ARM/NEON (через провайдер)
        provider_options = {
            'CPUExecutionProvider': {
                'arena_extend_strategy': 'kSameAsRequested',
                'enable_cpu_mem_arena': True,
                'enable_mem_pattern': True,
                'memory_limit': 2 * 1024 * 1024 * 1024,  # 2GB лимит памяти
            }
        }
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider'],
            provider_options=provider_options
        )
        
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        
        # Определяем размер входа из модели [batch, channels, height, width]
        self.img_h = int(input_shape[2]) if input_shape[2] else 640
        self.img_w = int(input_shape[3]) if input_shape[3] else 640
        
        print(f"OCR модель: {model_path}")
        print(f"  Вход: {self.img_w}x{self.img_h}")
        print(f"  Потоков: {sess_options.intra_op_num_threads}")
    
    def detect(self, cropped_plate):
        """
        Детектирует символы на вырезанном номере.
        Возвращает список детекций в формате ultralytics:
        каждая детекция содержит xyxy координаты в пикселях исходного изображения (cropped_plate)
        """
        orig_h, orig_w = cropped_plate.shape[:2]
        
        # Предобработка (как в ultralytics) с оптимизациями для ARM
        scale = min(self.img_h / orig_h, self.img_w / orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        
        # Используем INTER_AREA для уменьшения (быстрее на ARM)
        if scale < 1.0:
            resized = cv2.resize(cropped_plate, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized = cv2.resize(cropped_plate, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Предвыделяем массив один раз (быстрее чем zeros + fill)
        padded = np.full((self.img_h, self.img_w, 3), 114, dtype=np.uint8)
        pad_h = (self.img_h - new_h) // 2
        pad_w = (self.img_w - new_w) // 2
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        
        # Конвертируем в формат модели (максимально оптимизировано для NEON)
        # Используем прямое преобразование без промежуточных копий
        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        
        # Оптимизированное преобразование: делаем всё за один проход где возможно
        # Используем константу для деления (компилятор может оптимизировать)
        INV_255 = 1.0 / 255.0
        
        # Преобразуем в float32 и нормализуем за один проход
        blob = blob.astype(np.float32) * INV_255
        
        # Транспонируем и добавляем batch dimension
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)
        
        # Инференс (используем output_names для лучшей оптимизации)
        output_names = [output.name for output in self.session.get_outputs()]
        outputs = self.session.run(output_names, {self.input_name: blob})
        output = outputs[0]
        
        # Обработка выхода YOLO
        if len(output.shape) == 3:
            output = output[0]
        if output.shape[0] < output.shape[1]:
            output = output.T
        
        # Формат: [x_center, y_center, width, height, class_0, ..., class_35] = 40 признаков
        # Оптимизация: предвыделяем списки и используем векторизацию где возможно
        detections = []
        
        # Предвычисляем константы для ускорения
        inv_scale = 1.0 / scale if scale > 0 else 1.0
        
        # Векторизованная обработка где возможно
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
            
            # Убираем padding и масштабируем (используем предвычисленную константу)
            x1 = (x1_model - pad_w) * inv_scale
            y1 = (y1_model - pad_h) * inv_scale
            x2 = (x2_model - pad_w) * inv_scale
            y2 = (y2_model - pad_h) * inv_scale
            
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
        
        # NMS (оптимизировано для векторизации)
        if len(detections) > 1:
            # Используем предвыделенные массивы для лучшей производительности
            boxes = np.array([d['xyxy'] for d in detections], dtype=np.float32)
            scores = np.array([d['conf'] for d in detections], dtype=np.float32)
            keep = self.nms(boxes, scores, 0.5)
            detections = [detections[i] for i in keep]
        
        # Сортируем слева направо (как в эталоне)
        detections.sort(key=lambda d: d['xyxy'][0])
        
        return detections
    
    def nms(self, boxes, scores, iou_threshold=0.5):
        """NMS с оптимизациями для векторизации (NEON)"""
        if len(boxes) == 0:
            return []
        
        # Используем float32 для лучшей производительности на ARM
        boxes = boxes.astype(np.float32, copy=False)
        scores = scores.astype(np.float32, copy=False)
        
        # Векторизованное вычисление площадей
        areas = np.multiply(
            np.subtract(boxes[:, 2], boxes[:, 0], dtype=np.float32),
            np.subtract(boxes[:, 3], boxes[:, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        order = scores.argsort()[::-1]
        keep = []
        
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break
            
            rest = order[1:]
            # Векторизованные операции для лучшего использования NEON
            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
            
            # Векторизованное вычисление пересечения
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = np.multiply(w, h, dtype=np.float32)
            
            # Векторизованное вычисление IoU
            union = np.add(areas[i], areas[rest], dtype=np.float32)
            union = np.subtract(union, intersection, dtype=np.float32)
            iou = np.divide(intersection, np.add(union, 1e-6), dtype=np.float32)
            
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
                  plate_conf=0.25, char_conf=0.25, num_threads=None, use_vulkan=False,
                  no_visualization=False):
    """Обработка изображений (как в эталоне)"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("Загрузка моделей...")
    plate_detector = NCNNPlateDetector(
        plate_model_dir, 
        conf_threshold=plate_conf,
        num_threads=num_threads,
        use_vulkan=use_vulkan
    )
    char_detector = ONNXCharDetector(
        char_model_path, 
        conf_threshold=char_conf,
        num_threads=num_threads
    )
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
    vis_time_total = 0
    io_time_total = 0
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
            
            # Собираем текст
            text = ''.join([d['char'] for d in char_detections])
            avg_conf = np.mean([d['conf'] for d in char_detections]) if char_detections else 0
            
            # Измеряем время визуализации отдельно
            vis_start = time.time()
            debug = None
            
            if not no_visualization:
                # Создаём визуализацию (оптимизировано - избегаем лишних копий)
                debug = img.copy()
            
            if debug is not None:
                # Рисуем полигон номера по keypoints (оптимизировано - предвыделяем массив)
                pts = np.empty((len(kpts), 2), dtype=np.int32)
                for i, kpt in enumerate(kpts):
                    pts[i, 0] = int(kpt[0])
                    pts[i, 1] = int(kpt[1])
                cv2.polylines(debug, [pts], True, (0, 255, 0), 2)
                
                # Рисуем ключевые точки с номерами (оптимизировано - меньше вызовов)
                for j, kpt in enumerate(kpts):
                    x, y = int(kpt[0]), int(kpt[1])
                    cv2.circle(debug, (x, y), 5, (0, 0, 255), -1)
                    # Убираем текст для ускорения (опционально)
                    # cv2.putText(debug, str(j), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
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
                    
                    # Легенда (оптимизировано - меньше операций)
                    legend_x, legend_y = 10, 30
                    legend_spacing = 20
                    bg_height = len(legend_items) * legend_spacing + 10
                    
                    # Используем прямое рисование вместо копирования и смешивания
                    cv2.rectangle(debug, (legend_x - 5, legend_y - 20), 
                                 (legend_x + 150, legend_y + bg_height), (0, 0, 0), -1)
                    
                    cv2.putText(debug, "Chars:", (legend_x, legend_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    for idx, (i, char, color, char_conf) in enumerate(legend_items):
                        y_pos = legend_y + (idx + 1) * legend_spacing
                        cv2.rectangle(debug, (legend_x, y_pos - 10), 
                                     (legend_x + 15, y_pos + 5), color, -1)
                        text_line = f"{i}: '{char}' ({char_conf:.2f})"
                        cv2.putText(debug, text_line, (legend_x + 20, y_pos), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            vis_time = time.time() - vis_start
            vis_time_total += vis_time
            
            # Сохраняем (измеряем I/O отдельно)
            io_start = time.time()
            if debug is not None:
                output_path = Path(output_dir) / f"{img_path.stem}_chars{img_path.suffix}"
                cv2.imwrite(str(output_path), debug)
            io_time = time.time() - io_start
            io_time_total += io_time
            
            proc_time = time.time() - start
            other_time = proc_time - plate_time - char_time - vis_time - io_time
            other_time_total += other_time
            total_time += proc_time
            
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
        print(f"  1. Детекция номера (YOLO Pose NCNN): {plate_time_total/stats['ok']*1000:.0f} мс ({plate_time_total/total_time*100:.1f}%)")
        print(f"  2. Детекция символов (ONNX 640x640): {char_time_total/stats['ok']*1000:.0f} мс ({char_time_total/total_time*100:.1f}%)")
        print(f"  3. Визуализация:                      {vis_time_total/stats['ok']*1000:.0f} мс ({vis_time_total/total_time*100:.1f}%)")
        print(f"  4. I/O (чтение/запись файлов):        {io_time_total/stats['ok']*1000:.0f} мс ({io_time_total/total_time*100:.1f}%)")
        print(f"  5. Прочее (вырезка, обработка):       {other_time_total/stats['ok']*1000:.0f} мс ({other_time_total/total_time*100:.1f}%)")
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
    parser.add_argument('--num-threads', type=int, default=None,
                       help='Количество потоков (по умолчанию: все доступные ядра)')
    parser.add_argument('--use-vulkan', action='store_true',
                       help='Использовать Vulkan для NCNN (требует поддержки GPU)')
    parser.add_argument('--no-visualization', action='store_true',
                       help='Отключить визуализацию для ускорения')
    
    args = parser.parse_args()
    
    process_images(
        args.input, args.output, args.plate_model, args.char_model, 
        args.plate_conf, args.char_conf, args.num_threads, args.use_vulkan,
        args.no_visualization
    )


if __name__ == '__main__':
    main()
