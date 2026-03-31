# User configuration
sae_regex_pattern="gemma-scope-9b-pt-res-canonical"
model_name="gemma-2-9b"
llm_dtype="bfloat16"
cuda_device=1
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
    ".*layer_26.*(16k).canonical"
    ".*layer_27.*(16k).canonical"
    ".*layer_28.*(16k).canonical"
    ".*layer_29.*(16k).canonical"
    ".*layer_30.*(16k).canonical"
    ".*layer_31.*(16k).canonical"
    ".*layer_32.*(16k).canonical"
    ".*layer_33.*(16k).canonical"
    ".*layer_34.*(16k).canonical"
    ".*layer_35.*(16k).canonical"
    ".*layer_36.*(16k).canonical"
    ".*layer_37.*(16k).canonical"
    ".*layer_38.*(16k).canonical"
    ".*layer_39.*(16k).canonical"
    ".*layer_40.*(16k).canonical"
    ".*layer_41.*(16k).canonical"
)

for sae_block_pattern in "${sae_block_patterns[@]}"; do
    echo "Starting core eval for pattern ${sae_block_pattern}..."
    python sae_bench/evals/core/main.py "${sae_regex_pattern}" "${sae_block_pattern}" \
    --batch_size_prompts 16 \
    --n_eval_sparsity_variance_batches 2000 \
    --n_eval_reconstruction_batches 200 \
    --output_folder "gemma_2_9b/canonical/dense/eval_results/core" \
    --exclude_special_tokens_from_reconstruction --verbose --llm_dtype ${llm_dtype} || {
        echo "Core eval for pattern ${sae_block_pattern} failed, continuing to next pattern..."
        continue
    }
    echo "Completed core eval for pattern ${sae_block_pattern}"
done

declare -a sparsity_levels=(25 50)
# declare -a sparsity_levels=(75)

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
            --output_folder "gemma_2_9b/canonical/sparsity=${sparsity}/${pruning}/eval_results/core" \
            --exclude_special_tokens_from_reconstruction --verbose --llm_dtype ${llm_dtype} \
            --checkpoint "/research/nfs_khalili_17/gupte.31/reproduce/SAEBench/models/gemma_2_9b/sparsity=${sparsity}/${pruning}" || {
                echo "Core eval for pattern ${sae_block_pattern} failed, continuing to next pattern..."
                continue
            }
            echo "Completed core eval for pattern ${sae_block_pattern}"
        done
    done
done


