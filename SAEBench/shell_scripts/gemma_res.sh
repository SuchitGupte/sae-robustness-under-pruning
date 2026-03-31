# User configuration
sae_regex_pattern="gemma-scope-2b-pt-res-canonical"
model_name="gemma-2-2b"
model_name_it="gemma-2-2b-it"
llm_dtype="bfloat16"
cuda_device=0
# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

# Create array of patterns
declare -a sae_block_patterns=(
    ".*layer_0.*(16k).canonical"
    ".*layer_1.*(16k).canonical"
    ".*layer_2.*(16k).canonical"
    ".*layer_3.*(16k).canonical"
    ".*layer_4.*(16k).canonical"
    ".*layer_5.*(16k).canonical"
    ".*layer_6.*(16k).canonical"
    ".*layer_7.*(16k).canonical"
    ".*layer_8.*(16k).canonical"
    ".*layer_9.*(16k).canonical"
    ".*layer_10.*(16k).canonical"
    ".*layer_11.*(16k).canonical"
    ".*layer_12.*(16k).canonical"
    ".*layer_13.*(16k).canonical"
    ".*layer_14.*(16k).canonical"
    ".*layer_15.*(16k).canonical"
    ".*layer_16.*(16k).canonical"
    ".*layer_17.*(16k).canonical"
    ".*layer_18.*(16k).canonical"
    ".*layer_19.*(16k).canonical"
    ".*layer_20.*(16k).canonical"
    ".*layer_21.*(16k).canonical"
    ".*layer_22.*(16k).canonical"
    ".*layer_23.*(16k).canonical"
    ".*layer_24.*(16k).canonical"
    ".*layer_25.*(16k).canonical"
)

for sae_block_pattern in "${sae_block_patterns[@]}"; do
    echo "Starting core eval for pattern ${sae_block_pattern}..."
    python sae_bench/evals/core/main.py "${sae_regex_pattern}" "${sae_block_pattern}" \
    --batch_size_prompts 16 \
    --n_eval_sparsity_variance_batches 2000 \
    --n_eval_reconstruction_batches 200 \
    --output_folder "gemma_2_2b/canonical/dense/eval_results/core" \
    --exclude_special_tokens_from_reconstruction --verbose --llm_dtype ${llm_dtype} || {
        echo "Core eval for pattern ${sae_block_pattern} failed, continuing to next pattern..."
        continue
    }
    echo "Completed core eval for pattern ${sae_block_pattern}"
done


declare -a sparsity_levels=(10 25 40 50 60 75 90)

declare -a prunings=(
    "magnitude"
    "sparsegpt"
    "wanda"
)

for sparsity in "${sparsity_levels[@]}"; do
    for pruning in "${prunings[@]}"; do
        for sae_block_pattern in "${sae_block_patterns[@]}"; do
            echo "Starting core eval for pattern ${sae_block_pattern}..."
            python sae_bench/evals/core/main.py "${sae_regex_pattern}" "${sae_block_pattern}" \
            --batch_size_prompts 16 \
            --n_eval_sparsity_variance_batches 2000 \
            --n_eval_reconstruction_batches 200 \
            --output_folder "gemma_2_2b/canonical/sparsity=${sparsity}/${pruning}/eval_results/core" \
            --exclude_special_tokens_from_reconstruction --verbose --llm_dtype ${llm_dtype} \
            --checkpoint "/research/nfs_khalili_17/gupte.31/reproduce/SAEBench/models/gemma_2_2b/sparsity=${sparsity}/${pruning}" || {
                echo "Core eval for pattern ${sae_block_pattern} failed, continuing to next pattern..."
                continue
            }
            echo "Completed core eval for pattern ${sae_block_pattern}"
        done
    done
done
