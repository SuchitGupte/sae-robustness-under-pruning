#!/bin/bash

models=(
    "google/gemma-2-9b gemma_2_9b 0"
    "EleutherAI/pythia-70m pythia_70m 0"
)

uniform_sparsity=0.5
uniform_sparsity_percent=$(printf "%.0f" $(echo "$uniform_sparsity * 100" | bc))

run_uniform () {
    local model=$1
    local model_name=$2
    local method=$3

    echo "Running $method uniform: sparsity=$uniform_sparsity"
    python main.py \
        --model $model \
        --prune_method $method \
        --sparsity_ratio $uniform_sparsity \
        --sparsity_schedule uniform \
        --sparsity_type unstructured \
        --save "out/dynamic/${model_name}/uniform_${uniform_sparsity_percent}/${method}/"
}

for model_entry in "${models[@]}"
do
    model=$(echo $model_entry | awk '{print $1}')
    model_name=$(echo $model_entry | awk '{print $2}')
    cuda_device=$(echo $model_entry | awk '{print $3}')

    export CUDA_VISIBLE_DEVICES=$cuda_device

    echo "========================================"
    echo "Model: $model_name (CUDA $cuda_device)"
    echo "Running experiments for static sparsity $uniform_sparsity_percent with schedule uniform"
    echo "========================================"

    run_uniform "$model" "$model_name" "sparsegpt"
    run_uniform "$model" "$model_name" "wanda"
    run_uniform "$model" "$model_name" "magnitude"

    echo ""
    echo "Done with $model_name. Results in out/dynamic/${model_name}/uniform_${uniform_sparsity_percent}/"
    echo ""
done

echo "========================================"
echo "Uniform done. Starting schedule experiments."
echo "========================================"

s_max=0.75
s_min=0.25
max_percent=$(printf "%.0f" $(echo "$s_max * 100" | bc))
min_percent=$(printf "%.0f" $(echo "$s_min * 100" | bc))
schedules=(
    'linear'
    'cosine'
)

run_schedules () {
    local model=$1
    local model_name=$2
    local method=$3
    local schedule=$4

    echo "Running $method $schedule: min=$s_min max=$s_max"
    python main.py \
        --model $model \
        --prune_method $method \
        --sparsity_ratio $s_max \
        --sparsity_ratio_min $s_min \
        --sparsity_schedule $schedule \
        --sparsity_type unstructured \
        --save "out/dynamic/${model_name}/${schedule}_${min_percent}_${max_percent}/${method}/"
}

for model_entry in "${models[@]}"
do
    model=$(echo $model_entry | awk '{print $1}')
    model_name=$(echo $model_entry | awk '{print $2}')
    cuda_device=$(echo $model_entry | awk '{print $3}')

    export CUDA_VISIBLE_DEVICES=$cuda_device

    for schedule in "${schedules[@]}"
    do
        echo "========================================"
        echo "Model: $model_name (CUDA $cuda_device)"
        echo "Running experiments for dynamic sparsity $s_min -> $s_max with schedule $schedule"
        echo "========================================"

        run_schedules "$model" "$model_name" "sparsegpt" "$schedule"
        run_schedules "$model" "$model_name" "wanda"     "$schedule"
        run_schedules "$model" "$model_name" "magnitude" "$schedule"

    done

    echo ""
    echo "Done with $model_name. Results in out/dynamic/${model_name}/*/"
    echo ""
done
wait

echo "========================================"
echo "All models complete."
echo "========================================"
