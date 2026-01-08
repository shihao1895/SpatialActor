#!/bin/bash
nohup Xvfb :33 -screen 0 640x360x24 &
export DISPLAY=':33'
startxfce4 &

export COPPELIASIM_ROOT='YOUR_COPPELIASIM_ROOT'
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

tasks='all'
gpu_id=0
model_paths=(
YOUR_MODEL_PATH_1
YOUR_MODEL_PATH_2
...
)

for model_path in "${model_paths[@]}"; do
    echo ">>> Evaluating: $model_path"
    CUDA_VISIBLE_DEVICES=$gpu_id \
    python spatial_actor/eval.py \
        --eval-datafolder /YOUR_DATAFOLDER/test \
        --model-path $model_path \
        --tasks $tasks \
        --device 0 \
        --eval-episodes 25 \
        --log-name rlbench_all \
        --headless
    echo ">>> Finished: $model_path"
done

echo "========== All Done =========="
