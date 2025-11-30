#!/bin/bash
# Скрипт для push в GitHub

echo "=========================================="
echo "Push в GitHub репозиторий"
echo "=========================================="
echo ""

# Проверяем, есть ли remote
if git remote | grep -q "^origin$"; then
    echo "Remote 'origin' уже настроен:"
    git remote -v
    echo ""
    read -p "Использовать существующий remote? (y/n): " use_existing
    if [ "$use_existing" != "y" ]; then
        git remote remove origin
    fi
fi

# Если remote не настроен, просим URL
if ! git remote | grep -q "^origin$"; then
    echo "Создайте приватный репозиторий на GitHub:"
    echo "  1. Зайдите на https://github.com/new"
    echo "  2. Название: nn_pro (или любое другое)"
    echo "  3. Выберите Private"
    echo "  4. НЕ добавляйте README, .gitignore или лицензию"
    echo "  5. Нажмите 'Create repository'"
    echo ""
    read -p "Введите URL вашего репозитория (https://github.com/USERNAME/nn_pro.git): " repo_url
    
    if [ -z "$repo_url" ]; then
        echo "Ошибка: URL не указан"
        exit 1
    fi
    
    git remote add origin "$repo_url"
    echo "Remote добавлен: $repo_url"
fi

# Переименовываем ветку в main (если нужно)
current_branch=$(git branch --show-current)
if [ "$current_branch" != "main" ]; then
    git branch -M main
    echo "Ветка переименована в 'main'"
fi

# Push
echo ""
echo "Делаю push..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Успешно! Репозиторий загружен на GitHub"
    echo "Проверьте: $(git remote get-url origin | sed 's/\.git$//')"
else
    echo ""
    echo "❌ Ошибка при push"
    echo ""
    echo "Возможные причины:"
    echo "  1. Репозиторий не существует на GitHub"
    echo "  2. Нет прав доступа"
    echo "  3. Нужна аутентификация (токен доступа)"
    echo ""
    echo "Для HTTPS с токеном:"
    echo "  git remote set-url origin https://YOUR_TOKEN@github.com/USERNAME/nn_pro.git"
fi

