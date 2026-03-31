#!/bin/bash

model='google/gemma-2-2b'
sparsity_ratio=0.5
cuda_device=1

export CUDA_VISIBLE_DEVICES=$cuda_device

run_python_command () {
    local method=$1
    local save_dir=$2
    python main.py \
        --model $model \
        --prune_method $method \
        --sparsity_ratio $sparsity_ratio \
        --sparsity_type unstructured \
        --save $save_dir 
}


# for method in sparsegpt wanda magnitude; do
#     echo "========================================"
#     echo "method=$method | avg=$sparsity_ratio"
#     echo "========================================"
#     run_python_command $method $schedule "out/uniform/gemma_2_2b/$method/"
# done

for method in wanda_dynamic sparsegpt; do
    echo "========================================"
    echo "method=$method | avg=$sparsity_ratio"
    echo "========================================"
    run_python_command $method "out/uniform/gemma_2_2b/$method/"
done
