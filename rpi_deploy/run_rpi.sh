#!/bin/bash
# Run inference on Raspberry Pi (Two-Stage)

# Check requirements
# pip3 install -r requirements.txt

python3 run_rpi.py \
    --input input \
    --output output \
    --model model/lp_regressor.onnx

