#!/bin/bash
# Скрипт для push с использованием Personal Access Token

echo "=========================================="
echo "Push в GitHub с Personal Access Token"
echo "=========================================="
echo ""

echo "1. Создайте Personal Access Token на GitHub:"
echo "   - Зайдите на https://github.com/settings/tokens"
echo "   - Нажмите 'Generate new token' -> 'Generate new token (classic)'"
echo "   - Название: 'nn_pro_push'"
echo "   - Выберите scope: 'repo' (полный доступ к репозиториям)"
echo "   - Нажмите 'Generate token'"
echo "   - СКОПИРУЙТЕ токен (он показывается только один раз!)"
echo ""

read -p "Введите ваш GitHub username: " github_username
read -p "Введите название репозитория (по умолчанию: nn_pro): " repo_name
repo_name=${repo_name:-nn_pro}

read -sp "Вставьте Personal Access Token: " github_token
echo ""

if [ -z "$github_token" ]; then
    echo "Ошибка: токен не указан"
    exit 1
fi

# Проверяем, есть ли remote
if git remote | grep -q "^origin$"; then
    git remote remove origin
fi

# Добавляем remote с токеном
repo_url="https://${github_token}@github.com/${github_username}/${repo_name}.git"
git remote add origin "$repo_url"

# Переименовываем ветку в main
current_branch=$(git branch --show-current)
if [ "$current_branch" != "main" ]; then
    git branch -M main
fi

echo ""
echo "Делаю push..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Успешно! Репозиторий загружен на GitHub"
    echo "URL: https://github.com/${github_username}/${repo_name}"
    echo ""
    echo "⚠️  ВАЖНО: Токен сохранен в remote URL"
    echo "Для безопасности можно удалить токен из URL:"
    echo "  git remote set-url origin https://github.com/${github_username}/${repo_name}.git"
else
    echo ""
    echo "❌ Ошибка при push"
    echo "Проверьте:"
    echo "  1. Репозиторий создан на GitHub"
    echo "  2. Токен правильный и имеет права 'repo'"
    echo "  3. Username правильный"
fi

