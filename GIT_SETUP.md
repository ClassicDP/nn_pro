# Настройка Git и Push в GitHub

## 1. Настройте Git (если еще не настроено)

```bash
git config --global user.name "Ваше Имя"
git config --global user.email "your.email@example.com"
```

## 2. Создайте приватный репозиторий на GitHub

1. Зайдите на https://github.com
2. Нажмите "New repository"
3. Название: `nn_pro` (или любое другое)
4. Выберите **Private**
5. НЕ добавляйте README, .gitignore или лицензию
6. Нажмите "Create repository"

## 3. Добавьте remote и сделайте push

После создания репозитория GitHub покажет команды. Выполните:

```bash
cd /home/pi/Projects/nn_pro

# Добавьте remote (замените YOUR_USERNAME на ваш GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/nn_pro.git

# Или если используете SSH:
# git remote add origin git@github.com:YOUR_USERNAME/nn_pro.git

# Переименуйте ветку в main (если нужно)
git branch -M main

# Сделайте push
git push -u origin main
```

## 4. Проверка

После push проверьте на GitHub, что:
- ✅ Все файлы загружены
- ✅ Изображения (.jpg, .png) НЕ попали в репозиторий
- ✅ Модели (.onnx, .bin, .param) НЕ попали в репозиторий
- ✅ Только код и документация в репозитории

## Что исключено из репозитория (.gitignore):

- Все изображения (*.jpg, *.png, и т.д.)
- Модели (*.onnx, *.bin, *.param, *.pt)
- Директории input_images/ и output_chars/
- Виртуальное окружение (venv/)
- Временные файлы

