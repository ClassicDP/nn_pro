#!/bin/bash
# Простой скрипт для push с использованием токена из .github_token

set -e

TOKEN_FILE=".github_token"

if [ ! -f "$TOKEN_FILE" ]; then
    echo "❌ Файл $TOKEN_FILE не найден"
    echo ""
    echo "Создайте файл $TOKEN_FILE с вашим GitHub токеном:"
    echo "  echo 'YOUR_TOKEN' > $TOKEN_FILE"
    echo "  chmod 600 $TOKEN_FILE"
    exit 1
fi

GITHUB_TOKEN=$(cat "$TOKEN_FILE" | grep -v '^#' | grep -v '^$' | head -1)

if [ -z "$GITHUB_TOKEN" ]; then
    echo "❌ Токен не найден в $TOKEN_FILE"
    exit 1
fi

# Получаем URL репозитория
REPO_URL=$(git remote get-url origin 2>/dev/null || echo "")

if [ -z "$REPO_URL" ]; then
    echo "❌ Remote 'origin' не настроен"
    echo "Настройте: git remote add origin https://github.com/USERNAME/REPO.git"
    exit 1
fi

# Извлекаем username и repo из URL
if [[ "$REPO_URL" =~ github.com[:/]([^/]+)/([^/]+)\.git ]]; then
    USERNAME="${BASH_REMATCH[1]}"
    REPO="${BASH_REMATCH[2]}"
else
    echo "❌ Не удалось определить username/repo из URL: $REPO_URL"
    exit 1
fi

# Временно добавляем токен в URL
AUTH_URL="https://${GITHUB_TOKEN}@github.com/${USERNAME}/${REPO}.git"
git remote set-url origin "$AUTH_URL"

echo "Делаю push в https://github.com/${USERNAME}/${REPO}..."

# Push
if git push -u origin main 2>/dev/null || git push -u origin master 2>/dev/null; then
    echo "✅ Успешно!"
else
    # Пробуем определить текущую ветку
    CURRENT_BRANCH=$(git branch --show-current)
    git push -u origin "$CURRENT_BRANCH"
fi

# Удаляем токен из URL для безопасности
git remote set-url origin "https://github.com/${USERNAME}/${REPO}.git"

echo "Готово!"

