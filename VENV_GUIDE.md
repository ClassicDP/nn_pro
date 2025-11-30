# Руководство по работе с виртуальным окружением

## Что такое виртуальное окружение?

Виртуальное окружение (venv) - это изолированная среда Python, где можно устанавливать пакеты без влияния на системный Python.

## Как активировать виртуальное окружение?

**В терминале выполните:**
```bash
source venv/bin/activate
```

После активации вы увидите `(venv)` в начале строки терминала:
```
(venv) pi@raspberrypi:~/Projects/nn_pro $
```

## Как установить пакеты?

### Вариант 1: Активировать venv, затем установить
```bash
source venv/bin/activate
pip install rapidocr_onnxruntime
```

### Вариант 2: Использовать скрипт (рекомендуется)
```bash
./install_dependencies.sh rapidocr_onnxruntime
```

### Вариант 3: Использовать pip напрямую из venv (без активации)
```bash
venv/bin/pip install rapidocr_onnxruntime
```

## Как запускать Python скрипты?

### Вариант 1: Активировать venv, затем запустить
```bash
source venv/bin/activate
python detect_plates.py
```

### Вариант 2: Использовать Python из venv напрямую
```bash
venv/bin/python detect_plates.py
```

### Вариант 3: Использовать готовые скрипты
```bash
./run_detection.sh
```

## Как деактивировать виртуальное окружение?

```bash
deactivate
```

## Важные моменты

1. **Всегда активируйте venv перед установкой пакетов** - иначе получите ошибку `externally-managed-environment`

2. **Активируйте venv перед запуском Python скриптов** - иначе Python не найдет установленные пакеты

3. **Проверка активации**: Если в начале строки терминала есть `(venv)`, значит окружение активировано

4. **В каждом новом терминале** нужно заново активировать venv командой `source venv/bin/activate`

## Примеры использования

### Установка пакета
```bash
source venv/bin/activate
pip install rapidocr_onnxruntime
```

### Установка всех зависимостей из requirements.txt
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Запуск скрипта
```bash
source venv/bin/activate
python detect_plates.py input_images/test.jpg
```

### Проверка установленных пакетов
```bash
source venv/bin/activate
pip list
```




