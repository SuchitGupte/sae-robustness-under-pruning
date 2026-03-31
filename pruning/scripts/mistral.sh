#!/bin/bash

model="mistralai/Mistral-7B-v0.1"
sparsity_ratio=0.5
cuda_device=1


# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

# Define function to run python command
run_python_command () {
    python main.py \
    --model $model \
    --prune_method $1 \
    --sparsity_ratio $sparsity_ratio \
    --sparsity_type $2 \
    --save $3
}

# for sparsity in "${sparsity_levels[@]}"
# do
#     sparsity_percent=$(printf "%.0f" $(echo "$sparsity * 100" | bc))
#     echo "========================================"
#     echo "Running experiments for sparsity $sparsity"
#     echo "========================================"

#     ### WANDA ###
#     echo "Running wanda"
#     run_python_command "wanda" "$sparsity" "unstructured" \
#         "out/mistral_7b/sparsity=${sparsity_percent}/wanda/" \
#         "models/mistral_7b/sparsity=${sparsity_percent}/wanda/"

#     ### SparseGPT ###
#     echo "Running sparsegpt"
#     run_python_command "sparsegpt" "$sparsity" "unstructured" \
#         "out/mistral_7b/sparsity=${sparsity_percent}/sparsegpt/" \
#         "models/mistral_7b/sparsity=${sparsity_percent}/sparsegpt/"

#     ### Magnitude ###
#     echo "Running magnitude"
#     run_python_command "magnitude" "$sparsity" "unstructured" \
#         "out/mistral_7b/sparsity=${sparsity_percent}/magnitude/" \
#         "models/mistral_7b/sparsity=${sparsity_percent}/magnitude/"

#     echo "Finished sparsity $sparsity"
# done
echo "Running with sensitivity_wanda pruning method"
python main.py \
    --model $model \
    --prune_method sensitivity_wanda \
    --sparsity_ratio $sparsity_ratio \
    --sparsity_type unstructured \
    --save "out/sensitivity/mistral_7b/unstructured/sensitivity_wanda/"
echo "Finished sensitivity_wanda pruning method"
