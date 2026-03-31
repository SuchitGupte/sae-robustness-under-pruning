# User configuration
sae_regex_pattern="pythia-70m-deduped-res-sm"
model_name='EleutherAI/pythia-70m-deduped'
llm_dtype="float32"
cuda_device=3
# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

# Create array of patterns
declare -a sae_block_patterns=(
    "blocks.0.hook_resid_pre"
    "blocks.1.hook_resid_post"
    "blocks.2.hook_resid_post"
    "blocks.3.hook_resid_post"
    "blocks.4.hook_resid_post"
    "blocks.5.hook_resid_post"
)


for sae_block_pattern in "${sae_block_patterns[@]}"; do
    echo "Starting core eval for pattern ${sae_block_pattern}..."
    python sae_bench/evals/core/main.py "${sae_regex_pattern}" "${sae_block_pattern}" \
    --batch_size_prompts 16 \
    --n_eval_sparsity_variance_batches 2000 \
    --n_eval_reconstruction_batches 200 \
    --output_folder "pythia_70m/dense/eval_results/core" \
    --exclude_special_tokens_from_reconstruction --verbose --llm_dtype ${llm_dtype} || {
        echo "Core eval for pattern ${sae_block_pattern} failed, continuing to next pattern..."
        continue
    }
    echo "Completed core eval for pattern ${sae_block_pattern}"
done


declare -a sparsity_levels=(25 50 75)

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
            --output_folder "pythia_70m/sparsity=${sparsity}/${pruning}/eval_results/core" \
            --exclude_special_tokens_from_reconstruction --verbose --llm_dtype ${llm_dtype} \
            --checkpoint "/research/nfs_khalili_17/gupte.31/reproduce/SAEBench/models/pythia_70m/sparsity=${sparsity}/${pruning}" || {
                echo "Core eval for pattern ${sae_block_pattern} failed, continuing to next pattern..."
                continue
            }
            echo "Completed core eval for pattern ${sae_block_pattern}"
        done
    done
done
