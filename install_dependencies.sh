#!/bin/bash
# Скрипт для установки зависимостей в виртуальное окружение

# Активируем виртуальное окружение
source venv/bin/activate

# Обновляем pip
pip install --upgrade pip

# Устанавливаем зависимости из requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Установка зависимостей из requirements.txt..."
    pip install -r requirements.txt
else
    echo "Предупреждение: файл requirements.txt не найден"
fi

# Если переданы дополнительные пакеты, устанавливаем их
if [ $# -gt 0 ]; then
    echo "Установка дополнительных пакетов: $@"
    pip install "$@"
fi

echo "Готово! Виртуальное окружение активировано."
echo "Для активации вручную используйте: source venv/bin/activate"




