#!/usr/bin/env python3
"""
Сравнение производительности ONNX Runtime vs NCNN
"""
import os
import sys
import time
import subprocess
import argparse
from pathlib import Path

def run_onnx(input_dir, output_dir, threads=4):
    """Запуск ONNX Runtime"""
    print("\n" + "="*60)
    print("ONNX Runtime")
    print("="*60)
    
    cmd = [
        sys.executable, "run_rpi.py",
        "--model", "full",
        "--threads", str(threads),
        "--input", input_dir,
        "--output", output_dir
    ]
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    
    # Парсим результаты
    times = []
    for line in result.stdout.split('\n'):
        if 'Infer:' in line:
            try:
                ms = float(line.split('Infer:')[1].split('ms')[0].strip())
                times.append(ms)
            except:
                pass
    
    return times, elapsed, result.stdout

def run_ncnn(input_dir, output_dir, threads=4):
    """Запуск NCNN"""
    print("\n" + "="*60)
    print("NCNN")
    print("="*60)
    
    cmd = [
        sys.executable, "run_rpi_ncnn.py",
        "--model", "model_ncnn",
        "--threads", str(threads),
        "--input", input_dir,
        "--output", output_dir
    ]
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    
    # Парсим результаты
    times = []
    for line in result.stdout.split('\n'):
        if 'Infer:' in line:
            try:
                ms = float(line.split('Infer:')[1].split('ms')[0].strip())
                times.append(ms)
            except:
                pass
    
    return times, elapsed, result.stdout

def main():
    parser = argparse.ArgumentParser(description='Сравнение ONNX vs NCNN')
    parser.add_argument('--input', default='input', help='Input directory')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads')
    args = parser.parse_args()
    
    input_dir = args.input
    onnx_output = 'output_onnx_bench'
    ncnn_output = 'output_ncnn_bench'
    
    # Подсчет изображений
    image_files = list(Path(input_dir).glob('*.jpg')) + \
                  list(Path(input_dir).glob('*.jpeg')) + \
                  list(Path(input_dir).glob('*.png'))
    num_images = len(image_files)
    
    print(f"\n{'='*60}")
    print(f"Сравнение производительности ONNX Runtime vs NCNN")
    print(f"{'='*60}")
    print(f"Изображений: {num_images}")
    print(f"Потоков: {args.threads}")
    
    # ONNX Runtime
    onnx_times, onnx_total, onnx_output_text = run_onnx(input_dir, onnx_output, args.threads)
    
    # NCNN
    ncnn_times, ncnn_total, ncnn_output_text = run_ncnn(input_dir, ncnn_output, args.threads)
    
    # Статистика
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ")
    print("="*60)
    
    if onnx_times and ncnn_times:
        print(f"\nОбработано изображений: {len(onnx_times)}")
        
        print(f"\nONNX Runtime:")
        print(f"  Среднее время инференции: {sum(onnx_times)/len(onnx_times):.1f} ms")
        print(f"  Мин: {min(onnx_times):.1f} ms")
        print(f"  Макс: {max(onnx_times):.1f} ms")
        print(f"  Общее время: {onnx_total:.2f} сек")
        
        print(f"\nNCNN:")
        print(f"  Среднее время инференции: {sum(ncnn_times)/len(ncnn_times):.1f} ms")
        print(f"  Мин: {min(ncnn_times):.1f} ms")
        print(f"  Макс: {max(ncnn_times):.1f} ms")
        print(f"  Общее время: {ncnn_total:.2f} сек")
        
        avg_onnx = sum(onnx_times)/len(onnx_times)
        avg_ncnn = sum(ncnn_times)/len(ncnn_times)
        
        print(f"\nСравнение:")
        if avg_ncnn < avg_onnx:
            speedup = avg_onnx / avg_ncnn
            print(f"  NCNN быстрее на {speedup:.2f}x")
            print(f"  Ускорение: {(avg_onnx - avg_ncnn):.1f} ms ({((avg_onnx - avg_ncnn)/avg_onnx*100):.1f}%)")
        else:
            slowdown = avg_ncnn / avg_onnx
            print(f"  ONNX быстрее на {slowdown:.2f}x")
            print(f"  Замедление: {(avg_ncnn - avg_onnx):.1f} ms ({((avg_ncnn - avg_onnx)/avg_onnx*100):.1f}%)")
    
    print("\n" + "="*60)
    print("Готово!")
    print(f"Результаты сохранены в:")
    print(f"  ONNX: {onnx_output}/")
    print(f"  NCNN: {ncnn_output}/")

if __name__ == '__main__':
    main()


