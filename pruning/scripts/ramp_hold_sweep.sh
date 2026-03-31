#!/bin/bash

model='google/gemma-2-2b'
model_name='gemma-2-2b'
cuda_device=1

# Valid combos for ramp_hold where average sparsity = 0.5
# Constraint: min_s + 3*max_s = 2.0
# min=0.20, max=0.60 -> avg=0.50
# min=0.35, max=0.55 -> avg=0.50
# min=0.05, max=0.65 -> avg=0.50
combos=(
    "0.20 0.60"
    "0.35 0.55"
    "0.05 0.65"
)

export CUDA_VISIBLE_DEVICES=$cuda_device

run_ramp_hold () {
    local method=$1
    local s_min=$2
    local s_max=$3
    local min_percent=$(printf "%.0f" $(echo "$s_min * 100" | bc))
    local max_percent=$(printf "%.0f" $(echo "$s_max * 100" | bc))

    echo "Running $method ramp_hold: min=$s_min max=$s_max"
    python main.py \
        --model $model \
        --prune_method $method \
        --sparsity_ratio $s_max \
        --sparsity_ratio_min $s_min \
        --sparsity_schedule ramp_hold \
        --sparsity_type unstructured \
        --save "out/dynamic/${model_name}/ramp_hold_${min_percent}_${max_percent}/${method}/"
}

echo "========================================"
echo "ramp_hold sweep: ramp min->max in first half, hold max in second half"
echo "All combos satisfy average sparsity = 50%"
echo "========================================"

for combo in "${combos[@]}"
do
    s_min=$(echo $combo | awk '{print $1}')
    s_max=$(echo $combo | awk '{print $2}')

    echo ""
    echo "--- min=$s_min, max=$s_max ---"
    run_ramp_hold "sparsegpt" $s_min $s_max
    run_ramp_hold "wanda"     $s_min $s_max
    run_ramp_hold "magnitude" $s_min $s_max

done

echo ""
echo "========================================"
echo "Done. Results in out/dynamic/${model_name}/ramp_hold_*/"
echo "========================================"
