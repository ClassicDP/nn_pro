# Конвертация и использование модели на Raspberry Pi

## ✓ Готовые файлы

- `nanodet_320.onnx` — оригинальная ONNX модель
- `nanodet_320_simplified.onnx` — упрощенная ONNX модель (рекомендуется для конвертации)

## Шаг 1: Конвертация ONNX → NCNN

### Вариант А: На локальном компьютере (Linux/Mac)

```bash
# Скачать готовые инструменты
wget https://github.com/Tencent/ncnn/releases/latest/download/ncnn-YYYYMMDD-ubuntu-2204.zip
unzip ncnn-*.zip
cd ncnn-*/bin

# Конвертировать
./onnx2ncnn nanodet_320_simplified.onnx nanodet.param nanodet.bin

# Результат: nanodet.param + nanodet.bin
```

### Вариант Б: Собрать ncnn из исходников

```bash
git clone https://github.com/Tencent/ncnn.git
cd ncnn
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=OFF -DNCNN_BUILD_TOOLS=ON ..
make -j4

# Конвертировать
./tools/onnx/onnx2ncnn ../../nanodet_320_simplified.onnx nanodet.param nanodet.bin
```

### Вариант В: Онлайн конвертер

1. Загрузить `nanodet_320_simplified.onnx` на https://convertmodel.com/
2. Выбрать: ONNX → NCNN
3. Скачать `.param` и `.bin` файлы

---

## Шаг 2: Установка NCNN на Raspberry Pi

```bash
# Установка зависимостей
sudo apt-get update
sudo apt-get install -y cmake build-essential

# Скачать и собрать ncnn
git clone https://github.com/Tencent/ncnn.git
cd ncnn
mkdir build && cd build

# Для Raspberry Pi (ARM)
cmake -DCMAKE_BUILD_TYPE=Release \
      -DNCNN_VULKAN=OFF \
      -DNCNN_BUILD_EXAMPLES=ON \
      -DNCNN_OPENMP=ON \
      -DNCNN_ARM82=ON ..  # Для RPi 3/4

make -j4
sudo make install
sudo ldconfig
```

---

## Шаг 3: Инференс на Raspberry Pi

### Python (с ncnn-python)

```bash
# Установка ncnn Python wrapper
pip install ncnn
```

**inference_ncnn.py:**
```python
import ncnn
import cv2
import numpy as np

# Загрузка модели
net = ncnn.Net()
net.opt.use_vulkan_compute = False
net.opt.num_threads = 4  # Для RPi 4

# Загрузка весов
net.load_param("nanodet.param")
net.load_model("nanodet.bin")

# Подготовка изображения
img = cv2.imread("test.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (320, 320))

# Нормализация (ImageNet mean/std)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
img_norm = (img_resized / 255.0 - mean) / std
img_norm = img_norm.astype(np.float32)

# Создание ncnn Mat
mat_in = ncnn.Mat.from_pixels(img_resized, ncnn.Mat.PixelType.PIXEL_RGB, 320, 320)
mat_in.substract_mean_normalize([123.675, 116.28, 103.53], [58.395, 57.12, 57.375])  # ImageNet

# Инференс
ex = net.create_extractor()
ex.input("input", mat_in)

# Получение выходов (6 выходов: cls + reg для 3 уровней)
cls_8 = ncnn.Mat()
cls_16 = ncnn.Mat()
cls_32 = ncnn.Mat()
reg_8 = ncnn.Mat()
reg_16 = ncnn.Mat()
reg_32 = ncnn.Mat()

ex.extract("cls_pred_stride8", cls_8)
ex.extract("cls_pred_stride16", cls_16)
ex.extract("cls_pred_stride32", cls_32)
ex.extract("reg_pred_stride8", reg_8)
ex.extract("reg_pred_stride16", reg_16)
ex.extract("reg_pred_stride32", reg_32)

# Постобработка (NMS и декодирование bbox)
# ... (аналогично вашему коду в eval.py)
```

### C++ (рекомендуется для максимальной скорости)

**inference.cpp:**
```cpp
#include "net.h"
#include <opencv2/opencv.hpp>
#include <vector>

int main() {
    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.opt.num_threads = 4;
    
    // Загрузка модели
    net.load_param("nanodet.param");
    net.load_model("nanodet.bin");
    
    // Загрузка изображения
    cv::Mat img = cv::imread("test.jpg");
    cv::Mat img_rgb;
    cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
    
    // Resize
    cv::Mat img_resized;
    cv::resize(img_rgb, img_resized, cv::Size(320, 320));
    
    // Конвертация в ncnn::Mat
    ncnn::Mat in = ncnn::Mat::from_pixels(
        img_resized.data, 
        ncnn::Mat::PIXEL_RGB, 
        320, 320
    );
    
    // Нормализация (ImageNet)
    const float mean_vals[3] = {123.675f, 116.28f, 103.53f};
    const float std_vals[3] = {58.395f, 57.12f, 57.375f};
    in.substract_mean_normalize(mean_vals, std_vals);
    
    // Инференс
    ncnn::Extractor ex = net.create_extractor();
    ex.input("input", in);
    
    ncnn::Mat cls_8, cls_16, cls_32;
    ncnn::Mat reg_8, reg_16, reg_32;
    
    ex.extract("cls_pred_stride8", cls_8);
    ex.extract("cls_pred_stride16", cls_16);
    ex.extract("cls_pred_stride32", cls_32);
    ex.extract("reg_pred_stride8", reg_8);
    ex.extract("reg_pred_stride16", reg_16);
    ex.extract("reg_pred_stride32", reg_32);
    
    // Постобработка...
    
    return 0;
}
```

**Компиляция:**
```bash
g++ inference.cpp -o inference \
    `pkg-config --cflags --libs opencv4` \
    -lncnn -fopenmp -O3
```

---

## Параметры модели

- **Вход:** `input`, размер `[1, 3, 320, 320]`, формат RGB, нормализация ImageNet
- **Выходы:**
  - `cls_pred_stride8`: [1, 1, 40, 40] — классификация (stride 8)
  - `cls_pred_stride16`: [1, 1, 20, 20] — классификация (stride 16)
  - `cls_pred_stride32`: [1, 1, 10, 10] — классификация (stride 32)
  - `reg_pred_stride8`: [1, 4, 40, 40] — регрессия bbox (stride 8)
  - `reg_pred_stride16`: [1, 4, 20, 20] — регрессия bbox (stride 16)
  - `reg_pred_stride32`: [1, 4, 10, 10] — регрессия bbox (stride 32)

- **Классы:** 1 класс (Vehicle)
- **Порог уверенности:** 0.35 (рекомендуется)

---

## Производительность на Raspberry Pi

| Устройство | Частота | Время инференса | FPS |
|------------|---------|-----------------|-----|
| RPi 3B+ | 1.4 GHz | ~400-600 ms | 2-3 |
| RPi 4 (4GB) | 1.5 GHz | ~150-250 ms | 4-6 |
| RPi 5 | 2.4 GHz | ~80-120 ms | 8-12 |

**Оптимизация:**
- Использовать `num_threads=4` для RPi 4
- Включить ARM NEON оптимизации (`-DNCNN_ARM82=ON`)
- Уменьшить разрешение до 192x192 для ускорения

---

## Troubleshooting

**Ошибка: "undefined reference to cv::imread"**
- Установить OpenCV: `sudo apt-get install libopencv-dev`

**Ошибка: "libncnn.so not found"**
- Выполнить: `sudo ldconfig`

**Медленный инференс**
- Увеличить `num_threads`
- Уменьшить размер входа до 192x192
- Проверить, что не используется VULKAN (отключен для RPi)

---

## Полезные ссылки

- NCNN GitHub: https://github.com/Tencent/ncnn
- NCNN Wiki: https://github.com/Tencent/ncnn/wiki
- ncnn-python: https://github.com/Tencent/ncnn/tree/master/python

