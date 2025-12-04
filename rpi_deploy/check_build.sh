#!/bin/bash
# Проверка статуса сборки NCNN

echo "Проверка сборки NCNN..."
echo ""

if [ -f "$HOME/ncnn/build/tools/onnx/onnx2ncnn" ]; then
    echo "✓ onnx2ncnn готов!"
    ls -lh "$HOME/ncnn/build/tools/onnx/onnx2ncnn"
    echo ""
    echo "Можно запускать конвертацию:"
    echo "  cd rpi_deploy"
    echo "  ./convert_and_test.sh"
else
    echo "⏳ Сборка еще не завершена..."
    echo ""
    echo "Проверка процесса сборки:"
    ps aux | grep -E "(make|cmake)" | grep -v grep | head -3
    echo ""
    echo "Подождите завершения сборки или запустите вручную:"
    echo "  cd ~/ncnn/build"
    echo "  make -j4"
fi


