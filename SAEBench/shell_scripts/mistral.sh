# User configuration
sae_regex_pattern="mistral-7b-res-wg"
model_name="mistralai/Mistral-7B-v0.1"
llm_dtype="bfloat16"
cuda_device=1
# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

# Create array of patterns
declare -a sae_block_patterns=(
    "blocks.8.hook_resid_pre"
    "blocks.16.hook_resid_pre"
    "blocks.24.hook_resid_pre"
)

# for sae_block_pattern in "${sae_block_patterns[@]}"; do
#     echo "Starting core eval for pattern ${sae_block_pattern}..."
#     python sae_bench/evals/core/main.py "${sae_regex_pattern}" "${sae_block_pattern}" \
#     --batch_size_prompts 16 \
#     --n_eval_sparsity_variance_batches 2000 \
#     --n_eval_reconstruction_batches 200 \
#     --output_folder "mistral_7b/dense/eval_results/core" \
#     --exclude_special_tokens_from_reconstruction --verbose --llm_dtype ${llm_dtype} || {
#         echo "Core eval for pattern ${sae_block_pattern} failed, continuing to next pattern..."
#         continue
#     }
#     echo "Completed core eval for pattern ${sae_block_pattern}"
# done

declare -a sparsity_levels=(10 25 40 60 75 90)
# declare -a sparsity_levels=(50)
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
            --output_folder "mistral_7b/sparsity=${sparsity}/${pruning}/eval_results/core" \
            --exclude_special_tokens_from_reconstruction --verbose --llm_dtype ${llm_dtype} \
            --checkpoint "/research/nfs_khalili_17/gupte.31/reproduce/SAEBench/models/mistral_7b/sparsity=${sparsity}/${pruning}" || {
                echo "Core eval for pattern ${sae_block_pattern} failed, continuing to next pattern..."
                continue
            }
            echo "Completed core eval for pattern ${sae_block_pattern}"
        done
    done
done

