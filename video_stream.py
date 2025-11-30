"""
Обработка видеопотока с оптимизациями для Raspberry Pi
- Асинхронный захват кадров
- Пропуск кадров
- Простой трекинг между детекциями
"""
import cv2
import numpy as np
import onnxruntime as ort
import time
import argparse
from threading import Thread
from collections import deque


class FrameGrabber:
    """Асинхронный захват кадров в отдельном потоке"""
    
    def __init__(self, source=0, width=640, height=480):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Минимальный буфер
        
        self.frame = None
        self.stopped = False
        self.grabbed = False
        
    def start(self):
        Thread(target=self._update, daemon=True).start()
        return self
    
    def _update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.cap.read()
    
    def read(self):
        return self.frame
    
    def stop(self):
        self.stopped = True
        self.cap.release()


class SimpleTracker:
    """Простой трекер на основе IoU между кадрами"""
    
    def __init__(self, max_age=5):
        self.boxes = []
        self.scores = []
        self.age = 0
        self.max_age = max_age
    
    def update(self, boxes, scores):
        """Обновить трекер новыми детекциями"""
        self.boxes = boxes
        self.scores = scores
        self.age = 0
    
    def predict(self):
        """Получить текущие боксы (без новой детекции)"""
        self.age += 1
        if self.age > self.max_age:
            self.boxes = []
            self.scores = []
        return self.boxes, self.scores
    
    def is_valid(self):
        """Есть ли актуальные детекции"""
        return self.age <= self.max_age and len(self.boxes) > 0


class VideoProcessor:
    """Процессор видеопотока с оптимизациями"""
    
    def __init__(self, model_path, conf_threshold=0.25, detect_every=3):
        """
        Args:
            model_path: Путь к ONNX модели
            conf_threshold: Порог уверенности
            detect_every: Детектировать каждый N-й кадр (остальные - трекинг)
        """
        self.conf_threshold = conf_threshold
        self.detect_every = detect_every
        
        # Загружаем модель
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.img_size = input_shape[2] if len(input_shape) > 2 else 320
        
        self.tracker = SimpleTracker(max_age=detect_every + 2)
        self.frame_count = 0
        
        # Статистика
        self.detect_times = deque(maxlen=30)
        self.total_times = deque(maxlen=30)
        
        print(f"Модель: {model_path}")
        print(f"Вход: {self.img_size}x{self.img_size}")
        print(f"Детекция каждые {detect_every} кадров")
    
    def preprocess(self, image):
        """Быстрая предобработка"""
        h, w = image.shape[:2]
        scale = min(self.img_size / h, self.img_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        padded = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        padded.fill(114)
        pad_h = (self.img_size - new_h) // 2
        pad_w = (self.img_size - new_w) // 2
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        
        padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        padded = padded.astype(np.float32) / 255.0
        padded = np.transpose(padded, (2, 0, 1))
        padded = np.expand_dims(padded, axis=0)
        
        return padded, scale, (pad_w, pad_h)
    
    def postprocess(self, outputs, scale, pad, orig_shape):
        """Пост-обработка"""
        pad_w, pad_h = pad
        orig_h, orig_w = orig_shape
        
        predictions = outputs[0]
        if len(predictions.shape) == 3:
            predictions = predictions[0]
            if predictions.shape[0] < predictions.shape[1]:
                predictions = predictions.transpose(1, 0)
        
        num_det, num_feat = predictions.shape
        
        if num_feat > 5:
            boxes_raw = predictions[:, :4]
            obj_conf = predictions[:, 4:5]
            class_conf = predictions[:, 5:]
            
            if obj_conf.max() > 1.0:
                obj_conf = 1.0 / (1.0 + np.exp(-np.clip(obj_conf, -500, 500)))
            if class_conf.max() > 1.0:
                exp_conf = np.exp(class_conf - class_conf.max(axis=1, keepdims=True))
                class_conf = exp_conf / exp_conf.sum(axis=1, keepdims=True)
            
            scores = (obj_conf * class_conf.max(axis=1, keepdims=True)).flatten()
        else:
            boxes_raw = predictions[:, :4]
            scores = predictions[:, 4] if num_feat == 5 else np.ones(len(predictions))
        
        mask = scores > self.conf_threshold
        if not np.any(mask):
            return [], []
        
        boxes_raw = boxes_raw[mask]
        scores = scores[mask]
        
        # Конвертируем координаты
        x1 = (boxes_raw[:, 0] - boxes_raw[:, 2] / 2 - pad_w) / scale
        y1 = (boxes_raw[:, 1] - boxes_raw[:, 3] / 2 - pad_h) / scale
        x2 = (boxes_raw[:, 0] + boxes_raw[:, 2] / 2 - pad_w) / scale
        y2 = (boxes_raw[:, 1] + boxes_raw[:, 3] / 2 - pad_h) / scale
        
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)
        
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        
        # Простой NMS
        keep = self.nms(boxes, scores)
        
        return boxes[keep].tolist(), scores[keep].tolist()
    
    def nms(self, boxes, scores, iou_threshold=0.45):
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
            
            order = rest[iou <= iou_threshold]
        
        return np.array(keep)
    
    def detect(self, image):
        """Полная детекция"""
        orig_shape = image.shape[:2]
        preprocessed, scale, pad = self.preprocess(image)
        
        start = time.time()
        outputs = self.session.run(None, {self.input_name: preprocessed})
        detect_time = time.time() - start
        
        boxes, scores = self.postprocess(outputs, scale, pad, orig_shape)
        self.detect_times.append(detect_time)
        
        return boxes, scores
    
    def process_frame(self, frame):
        """
        Обработка кадра с оптимизацией:
        - Каждый N-й кадр: полная детекция
        - Остальные кадры: используем предыдущие детекции (трекинг)
        """
        start = time.time()
        
        self.frame_count += 1
        
        if self.frame_count % self.detect_every == 0:
            # Полная детекция
            boxes, scores = self.detect(frame)
            self.tracker.update(boxes, scores)
            is_detection = True
        else:
            # Трекинг (используем предыдущие боксы)
            boxes, scores = self.tracker.predict()
            is_detection = False
        
        self.total_times.append(time.time() - start)
        
        return boxes, scores, is_detection
    
    def draw(self, frame, boxes, scores, is_detection=True):
        """Рисуем результаты"""
        color = (0, 255, 0) if is_detection else (0, 255, 255)  # Зелёный/жёлтый
        
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{score:.2f}"
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Статистика
        if len(self.total_times) > 0:
            avg_total = np.mean(self.total_times) * 1000
            fps = 1000 / avg_total if avg_total > 0 else 0
            
            if len(self.detect_times) > 0:
                avg_detect = np.mean(self.detect_times) * 1000
                cv2.putText(frame, f"Detect: {avg_detect:.0f}ms", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            mode = "DETECT" if is_detection else "TRACK"
            cv2.putText(frame, mode, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame
    
    def get_stats(self):
        """Получить статистику"""
        if len(self.total_times) == 0:
            return {}
        
        avg_total = np.mean(self.total_times) * 1000
        avg_detect = np.mean(self.detect_times) * 1000 if len(self.detect_times) > 0 else 0
        
        return {
            'fps': 1000 / avg_total if avg_total > 0 else 0,
            'avg_frame_ms': avg_total,
            'avg_detect_ms': avg_detect,
            'detect_every': self.detect_every
        }


def run_video(source, model_path, detect_every=3, show=True, save_path=None):
    """Запуск обработки видео"""
    
    # Определяем источник
    if source.isdigit():
        source = int(source)
    
    # Асинхронный захват кадров
    print(f"Открываем источник: {source}")
    grabber = FrameGrabber(source, width=640, height=480)
    grabber.start()
    
    # Ждём первый кадр
    time.sleep(0.5)
    
    # Процессор
    processor = VideoProcessor(model_path, detect_every=detect_every)
    
    # Видеозапись
    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(save_path, fourcc, 20, (640, 480))
    
    print("\nНажмите 'q' для выхода\n")
    
    try:
        while True:
            frame = grabber.read()
            if frame is None:
                continue
            
            # Обработка
            boxes, scores, is_detection = processor.process_frame(frame)
            
            # Визуализация
            result = processor.draw(frame.copy(), boxes, scores, is_detection)
            
            if writer:
                writer.write(result)
            
            if show:
                cv2.imshow('License Plate Detection', result)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    except KeyboardInterrupt:
        pass
    
    finally:
        # Итоговая статистика
        stats = processor.get_stats()
        print("\n" + "="*50)
        print("СТАТИСТИКА ВИДЕОПОТОКА")
        print("="*50)
        print(f"Эффективный FPS: {stats.get('fps', 0):.1f}")
        print(f"Среднее время кадра: {stats.get('avg_frame_ms', 0):.0f} мс")
        print(f"Среднее время детекции: {stats.get('avg_detect_ms', 0):.0f} мс")
        print(f"Детекция каждые {stats.get('detect_every', 0)} кадров")
        print("="*50)
        
        grabber.stop()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Обработка видеопотока')
    parser.add_argument('--source', type=str, default='0',
                        help='Источник видео: 0 для камеры, путь к файлу, или RTSP URL')
    parser.add_argument('--model', type=str, default='best.onnx', help='Путь к модели')
    parser.add_argument('--detect-every', type=int, default=3,
                        help='Детектировать каждый N-й кадр (остальные - трекинг)')
    parser.add_argument('--no-show', action='store_true', help='Не показывать окно')
    parser.add_argument('--save', type=str, default=None, help='Сохранить видео в файл')
    
    args = parser.parse_args()
    
    run_video(
        source=args.source,
        model_path=args.model,
        detect_every=args.detect_every,
        show=not args.no_show,
        save_path=args.save
    )


if __name__ == '__main__':
    main()





