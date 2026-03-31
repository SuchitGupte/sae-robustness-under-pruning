model='EleutherAI/pythia-70m-deduped'
sparsity_levels=(0.25 0.50 0.75)
cuda_device=3


# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

# Define function to run python command
run_python_command () {
    python main.py \
    --model $model \
    --prune_method $1 \
    --sparsity_ratio $2 \
    --sparsity_type $3 \
    --save $4 \
    --save_model $5
}

for sparsity in "${sparsity_levels[@]}"
do
    sparsity_percent=$(printf "%.0f" $(echo "$sparsity * 100" | bc))
    echo "========================================"
    echo "Running experiments for sparsity $sparsity"
    echo "========================================"

    ### WANDA ###
    echo "Running wanda"
    run_python_command "wanda" "$sparsity" "unstructured" \
        "out/pythia_70m/sparsity=${sparsity_percent}/wanda/" \
        "models/pythia_70m/sparsity=${sparsity_percent}/wanda/"

    ### SparseGPT ###
    echo "Running sparsegpt"
    run_python_command "sparsegpt" "$sparsity" "unstructured" \
        "out/pythia_70m/sparsity=${sparsity_percent}/sparsegpt/" \
        "models/pythia_70m/sparsity=${sparsity_percent}/sparsegpt/"

    ### Magnitude ###
    echo "Running magnitude"
    run_python_command "magnitude" "$sparsity" "unstructured" \
        "out/pythia_70m/sparsity=${sparsity_percent}/magnitude/" \
        "models/pythia_70m/sparsity=${sparsity_percent}/magnitude/"

    echo "Finished sparsity $sparsity"
done



