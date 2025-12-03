# Быстрый старт с NCNN

NCNN может быть **в 2-3 раза быстрее** чем ONNX Runtime на Raspberry Pi!

## Шаг 1: Установка

```bash
cd rpi_deploy
source .venv/bin/activate
pip install ncnn
```

## Шаг 2: Конвертация модели

Для конвертации ONNX → NCNN нужна утилита `onnx2ncnn`. 

### Вариант A: Использовать готовую модель (если есть)

Если у вас уже есть NCNN модель, просто скопируйте файлы `.param` и `.bin` в директорию `model_ncnn/`.

### Вариант B: Конвертировать самостоятельно

Для конвертации нужно собрать NCNN из исходников:

```bash
# Установка зависимостей
sudo apt-get update
sudo apt-get install build-essential cmake git

# Сборка NCNN
git clone https://github.com/Tencent/ncnn.git
cd ncnn
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

# Конвертация
cd ../../rpi_deploy
python3 convert_to_ncnn.py \
    --input model/lp_regressor.onnx \
    --output model_ncnn \
    --onnx2ncnn ~/ncnn/build/tools/onnx/onnx2ncnn
```

## Шаг 3: Использование

```bash
python3 run_rpi_ncnn.py --model model_ncnn --threads 4
```

## Ожидаемая производительность

- **ONNX**: ~100-110ms на инференцию
- **NCNN**: ~30-50ms на инференцию (в 2-3 раза быстрее!)

## Примечания

- NCNN оптимизирован для ARM процессоров
- Модель NCNN обычно меньше по размеру (~3MB vs 5.6MB)
- Для максимальной производительности рекомендуется собрать NCNN из исходников

