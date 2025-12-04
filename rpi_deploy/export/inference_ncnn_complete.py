"""
Полный пример инференса NanoDet на Raspberry Pi с NCNN
Требует: pip install ncnn opencv-python numpy
"""
import ncnn
import cv2
import numpy as np
import time
from pathlib import Path


class NanoDetNCNN:
    """Детектор транспорта на базе NCNN для Raspberry Pi"""
    
    def __init__(self, param_path, bin_path, input_size=320, conf_threshold=0.35, num_threads=4):
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        
        # Инициализация сети
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = num_threads
        
        # Загрузка модели
        ret_param = self.net.load_param(param_path)
        ret_model = self.net.load_model(bin_path)
        
        if ret_param != 0 or ret_model != 0:
            raise RuntimeError(f"Failed to load model: param={ret_param}, model={ret_model}")
        
        print(f"✓ Model loaded: {param_path}")
        print(f"  - Input size: {input_size}x{input_size}")
        print(f"  - Conf threshold: {conf_threshold}")
        print(f"  - Threads: {num_threads}")
    
    def preprocess(self, img):
        """Предобработка изображения"""
        # Resize
        img_resized = cv2.resize(img, (self.input_size, self.input_size))
        
        # Конвертация BGR -> RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Создание ncnn Mat с нормализацией ImageNet
        mat_in = ncnn.Mat.from_pixels(
            img_rgb, 
            ncnn.Mat.PixelType.PIXEL_RGB, 
            self.input_size, 
            self.input_size
        )
        
        # Нормализация: (x - mean) / std
        # ImageNet mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
        # NCNN использует: pixel = (pixel - mean) * (1/std)
        # mean_vals = mean * 255, norm_vals = 1/(std*255)
        mean_vals = [123.675, 116.28, 103.53]  # [0.485*255, 0.456*255, 0.406*255]
        norm_vals = [0.01712475, 0.0175, 0.01742919]  # [1/(0.229*255), ...]
        
        mat_in.substract_mean_normalize(mean_vals, norm_vals)
        
        return mat_in
    
    def decode_predictions(self, cls_preds, reg_preds, strides, orig_shape):
        """Декодирование предсказаний в bounding boxes"""
        detections = []
        orig_h, orig_w = orig_shape
        
        for cls_pred, reg_pred, stride in zip(cls_preds, reg_preds, strides):
            h, w = cls_pred.shape[1:3]
            
            # Применяем сигмоиду к классам
            scores = 1.0 / (1.0 + np.exp(-cls_pred))  # sigmoid
            
            # Проходим по всем ячейкам
            for i in range(h):
                for j in range(w):
                    score = scores[0, i, j, 0]
                    
                    if score < self.conf_threshold:
                        continue
                    
                    # Декодирование bbox
                    # reg_pred: [l, t, r, b] - расстояния от центра ячейки
                    l, t, r, b = reg_pred[0, i, j, :]
                    
                    # Центр ячейки в исходных координатах
                    cx = (j + 0.5) * stride
                    cy = (i + 0.5) * stride
                    
                    # Координаты bbox
                    x1 = (cx - l) / self.input_size * orig_w
                    y1 = (cy - t) / self.input_size * orig_h
                    x2 = (cx + r) / self.input_size * orig_w
                    y2 = (cy + b) / self.input_size * orig_h
                    
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
        orig_h, orig_w = img.shape[:2]
        
        # Предобработка
        mat_in = self.preprocess(img)
        
        # Инференс
        ex = self.net.create_extractor()
        ex.input("input", mat_in)
        
        # Извлечение выходов
        cls_8 = ex.extract("cls_pred_stride8")[1]
        cls_16 = ex.extract("cls_pred_stride16")[1]
        cls_32 = ex.extract("cls_pred_stride32")[1]
        reg_8 = ex.extract("reg_pred_stride8")[1]
        reg_16 = ex.extract("reg_pred_stride16")[1]
        reg_32 = ex.extract("reg_pred_stride32")[1]
        
        # Конвертация в numpy
        cls_preds = [
            np.array(cls_8).reshape(1, cls_8.h, cls_8.w, cls_8.c),
            np.array(cls_16).reshape(1, cls_16.h, cls_16.w, cls_16.c),
            np.array(cls_32).reshape(1, cls_32.h, cls_32.w, cls_32.c)
        ]
        reg_preds = [
            np.array(reg_8).reshape(1, reg_8.h, reg_8.w, reg_8.c),
            np.array(reg_16).reshape(1, reg_16.h, reg_16.w, reg_16.c),
            np.array(reg_32).reshape(1, reg_32.h, reg_32.w, reg_32.c)
        ]
        
        strides = [8, 16, 32]
        
        # Декодирование
        detections = self.decode_predictions(cls_preds, reg_preds, strides, (orig_h, orig_w))
        
        # NMS
        detections = self.nms(detections, iou_threshold=0.5)
        
        return detections
    
    def draw_detections(self, img, detections):
        """Отрисовка детекций на изображении"""
        img_draw = img.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            score = det['score']
            
            # Бокс
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Текст
            text = f"Vehicle {score:.2f}"
            cv2.putText(img_draw, text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return img_draw


def main():
    """Пример использования"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--param', type=str, default='nanodet.param', help='Path to .param file')
    parser.add_argument('--bin', type=str, default='nanodet.bin', help='Path to .bin file')
    parser.add_argument('--image', type=str, required=True, help='Path to test image')
    parser.add_argument('--size', type=int, default=320, help='Input size')
    parser.add_argument('--conf', type=float, default=0.35, help='Confidence threshold')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads')
    parser.add_argument('--output', type=str, default='result.jpg', help='Output image path')
    
    args = parser.parse_args()
    
    # Инициализация детектора
    detector = NanoDetNCNN(
        param_path=args.param,
        bin_path=args.bin,
        input_size=args.size,
        conf_threshold=args.conf,
        num_threads=args.threads
    )
    
    # Загрузка изображения
    img = cv2.imread(args.image)
    if img is None:
        print(f"✗ Failed to load image: {args.image}")
        return
    
    print(f"✓ Image loaded: {img.shape}")
    
    # Детекция
    print("Running detection...")
    start = time.time()
    detections = detector.detect(img)
    elapsed = time.time() - start
    
    print(f"✓ Detection done in {elapsed*1000:.1f} ms ({1/elapsed:.1f} FPS)")
    print(f"  Found {len(detections)} vehicles")
    
    # Отрисовка
    img_result = detector.draw_detections(img, detections)
    cv2.imwrite(args.output, img_result)
    print(f"✓ Result saved to: {args.output}")


if __name__ == '__main__':
    main()

