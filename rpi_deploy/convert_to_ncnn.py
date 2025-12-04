#!/usr/bin/env python3
"""
Конвертация ONNX модели в NCNN формат для лучшей производительности на Raspberry Pi
Требуется: установленный NCNN с утилитой onnx2ncnn
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

def find_onnx2ncnn():
    """Поиск утилиты onnx2ncnn"""
    # Возможные пути
    possible_paths = [
        'onnx2ncnn',
        '/usr/local/bin/onnx2ncnn',
        '/usr/bin/onnx2ncnn',
        '~/ncnn/build/tools/onnx/onnx2ncnn',
        os.path.expanduser('~/ncnn/build/tools/onnx/onnx2ncnn'),
    ]
    
    for path in possible_paths:
        try:
            result = subprocess.run([path, '--version'], 
                                   capture_output=True, 
                                   timeout=5)
            if result.returncode == 0:
                return path
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    
    return None

def convert_onnx_to_ncnn(onnx_path, output_dir, onnx2ncnn_path=None):
    """Конвертация ONNX модели в NCNN"""
    if not os.path.exists(onnx_path):
        print(f"Ошибка: Файл {onnx_path} не найден")
        return False
    
    # Находим утилиту
    if onnx2ncnn_path is None:
        onnx2ncnn_path = find_onnx2ncnn()
    
    if onnx2ncnn_path is None:
        print("Ошибка: Утилита onnx2ncnn не найдена!")
        print("\nУстановка NCNN:")
        print("1. Скачайте NCNN: https://github.com/Tencent/ncnn")
        print("2. Соберите:")
        print("   cd ncnn")
        print("   mkdir build && cd build")
        print("   cmake ..")
        print("   make -j4")
        print("3. Утилита будет в: build/tools/onnx/onnx2ncnn")
        return False
    
    # Создаем выходную директорию
    os.makedirs(output_dir, exist_ok=True)
    
    # Имена выходных файлов
    model_name = Path(onnx_path).stem
    param_path = os.path.join(output_dir, f"{model_name}.param")
    bin_path = os.path.join(output_dir, f"{model_name}.bin")
    
    print(f"Конвертация {onnx_path} -> NCNN...")
    print(f"  Параметры: {param_path}")
    print(f"  Веса: {bin_path}")
    
    # Запускаем конвертацию
    try:
        result = subprocess.run(
            [onnx2ncnn_path, onnx_path, param_path, bin_path],
            capture_output=True,
            text=True,
            timeout=300  # 5 минут максимум
        )
        
        if result.returncode != 0:
            print(f"Ошибка конвертации:")
            print(result.stderr)
            return False
        
        if os.path.exists(param_path) and os.path.exists(bin_path):
            param_size = os.path.getsize(param_path) / 1024
            bin_size = os.path.getsize(bin_path) / (1024 * 1024)
            onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
            
            print(f"\n✓ Конвертация успешна!")
            print(f"  ONNX размер: {onnx_size:.2f} MB")
            print(f"  NCNN param: {param_size:.2f} KB")
            print(f"  NCNN bin: {bin_size:.2f} MB")
            print(f"  Сжатие: {onnx_size/bin_size:.1f}x")
            return True
        else:
            print("Ошибка: Файлы не созданы")
            return False
            
    except subprocess.TimeoutExpired:
        print("Ошибка: Таймаут конвертации")
        return False
    except Exception as e:
        print(f"Ошибка: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Конвертация ONNX в NCNN')
    parser.add_argument('--input', required=True, help='Путь к ONNX модели')
    parser.add_argument('--output', default='model_ncnn', help='Выходная директория')
    parser.add_argument('--onnx2ncnn', help='Путь к утилите onnx2ncnn (автопоиск если не указан)')
    args = parser.parse_args()
    
    success = convert_onnx_to_ncnn(args.input, args.output, args.onnx2ncnn)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()


