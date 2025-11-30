"""
Тестирование RapidOCR на изображениях номерных знаков
Использует rapidocr_onnxruntime для распознавания текста
"""
import os
import cv2
import numpy as np
from rapidocr import RapidOCR
import argparse
import time
from pathlib import Path


def process_images_with_ocr(input_dir, output_dir, use_detection=True):
    """
    Обрабатывает изображения с помощью RapidOCR
    
    Args:
        input_dir: Каталог с входными изображениями
        output_dir: Каталог для сохранения результатов
        use_detection: Если True, использует детекцию текста перед распознаванием
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Инициализируем RapidOCR
    print("Инициализация RapidOCR...")
    ocr_engine = RapidOCR()
    print("RapidOCR готов!\n")
    
    # Получаем список изображений
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))
        image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"Не найдено изображений в: {input_dir}")
        return
    
    print(f"Найдено изображений: {len(image_files)}")
    print(f"Режим детекции: {'Включен' if use_detection else 'Выключен'}\n")
    
    total_time = 0
    total_texts = 0
    processed = 0
    
    for img_path in image_files:
        try:
            # Загружаем изображение
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Не удалось загрузить: {img_path.name}")
                continue
            
            # Распознавание текста
            start = time.time()
            result = ocr_engine(str(img_path))
            proc_time = time.time() - start
            
            total_time += proc_time
            processed += 1
            
            # Обрабатываем результаты (новый формат RapidOCROutput)
            detected_texts = []
            
            if result and len(result.boxes) > 0:
                # result.boxes - координаты bounding box
                # result.txts - распознанные тексты
                # result.scores - уверенность
                for i in range(len(result.boxes)):
                    box = result.boxes[i]
                    text = result.txts[i] if i < len(result.txts) else ""
                    confidence = result.scores[i] if i < len(result.scores) else 1.0
                    
                    detected_texts.append({
                        'text': text,
                        'confidence': confidence,
                        'box': box
                    })
                    total_texts += 1
            
            # Рисуем результаты на изображении
            result_image = image.copy()
            
            for item in detected_texts:
                box = item['box']
                text = item['text']
                conf = item['confidence']
                
                # Преобразуем координаты в numpy array
                # box имеет формат [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                if isinstance(box, (list, np.ndarray)) and len(box) > 0:
                    pts = np.array(box, dtype=np.int32)
                    
                    # Рисуем полигон
                    cv2.polylines(result_image, [pts], True, (0, 255, 0), 2)
                    
                    # Полупрозрачная заливка
                    overlay = result_image.copy()
                    cv2.fillPoly(overlay, [pts], (0, 255, 0))
                    cv2.addWeighted(overlay, 0.2, result_image, 0.8, 0, result_image)
                    
                    # Текст и уверенность
                    label = f"{text} ({conf:.2f})"
                    x, y = int(pts[0][0]), int(pts[0][1])
                    
                    # Фон для текста
                    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(result_image, (x, y - th - bl - 5), 
                                 (x + tw, y), (0, 255, 0), -1)
                    cv2.putText(result_image, label, (x, y - bl - 2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Добавляем общую информацию
            info_text = f"Texts: {len(detected_texts)} | Time: {proc_time*1000:.0f}ms"
            cv2.putText(result_image, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Сохраняем результат
            output_path = Path(output_dir) / f"{img_path.stem}_ocr{img_path.suffix}"
            cv2.imwrite(str(output_path), result_image)
            
            # Выводим распознанные тексты
            if detected_texts:
                texts_str = " | ".join([f"{item['text']}({item['confidence']:.2f})" 
                                      for item in detected_texts])
                print(f"[{processed}/{len(image_files)}] {img_path.name}: "
                      f"{len(detected_texts)} текстов, {proc_time*1000:.0f} мс")
                print(f"  Тексты: {texts_str}")
            else:
                print(f"[{processed}/{len(image_files)}] {img_path.name}: "
                      f"текст не найден, {proc_time*1000:.0f} мс")
            
        except Exception as e:
            print(f"Ошибка при обработке {img_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Статистика
    print("\n" + "="*50)
    print("СТАТИСТИКА OCR")
    print("="*50)
    print(f"Обработано: {processed}/{len(image_files)}")
    print(f"Общее время: {total_time:.2f}с")
    if processed > 0:
        fps = processed / total_time
        ms_per_img = total_time / processed * 1000
        print(f"Время на изображение: {ms_per_img:.0f} мс")
        print(f"FPS: {fps:.1f}")
        print(f"Всего текстов найдено: {total_texts}")
        print(f"Среднее текстов на изображение: {total_texts/processed:.1f}")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(
        description='Тестирование RapidOCR на изображениях номерных знаков'
    )
    parser.add_argument(
        '--input', 
        type=str, 
        default='./input_images', 
        help='Входной каталог с изображениями'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='./output_ocr', 
        help='Выходной каталог для результатов'
    )
    parser.add_argument(
        '--no-detection', 
        action='store_true',
        help='Отключить детекцию текста (только распознавание)'
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input):
        print(f"Ошибка: каталог не существует: {args.input}")
        return
    
    process_images_with_ocr(
        args.input, 
        args.output, 
        use_detection=not args.no_detection
    )


if __name__ == '__main__':
    main()

