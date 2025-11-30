# Настройка SSH для GitHub (если хотите использовать SSH ключи с десктопа)

## Вариант 1: Скопировать SSH ключ с десктопа на Raspberry Pi

### На десктопе:
1. Найдите ваш SSH ключ:
   ```bash
   cat ~/.ssh/id_rsa.pub
   # или
   cat ~/.ssh/id_ed25519.pub
   ```

2. Скопируйте содержимое ключа

### На Raspberry Pi:
1. Создайте директорию для SSH ключей:
   ```bash
   mkdir -p ~/.ssh
   chmod 700 ~/.ssh
   ```

2. Добавьте публичный ключ в authorized_keys (если нужен доступ к Pi) или создайте новый ключ:
   ```bash
   ssh-keygen -t ed25519 -C "pi@raspberrypi"
   ```

3. Скопируйте публичный ключ:
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```

4. Добавьте ключ на GitHub:
   - Зайдите на https://github.com/settings/keys
   - Нажмите "New SSH key"
   - Вставьте публичный ключ
   - Сохраните

5. Используйте SSH URL для push:
   ```bash
   git remote add origin git@github.com:USERNAME/nn_pro.git
   git push -u origin main
   ```

## Вариант 2: Использовать Personal Access Token (проще)

См. инструкцию в `push_with_token.sh`

