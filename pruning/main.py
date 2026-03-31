import argparse
import os 
import numpy as np
import torch
from modeling_llama import LlamaForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTNeoXForCausalLM
from importlib.metadata import version
from lib.dynamic_pruning import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity
from lib.eval import eval_ppl, eval_zero_shot
from lora_ft.evaluate_ppl import evaluate_ppl
from transformers import pipeline

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model_name, cache_dir="llm_weights"):
    if 'llama' in model_name.lower():
        model = LlamaForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            cache_dir=cache_dir, 
            low_cpu_mem_usage=True, 
            device_map="auto"
        )
    elif 'pythia' in model_name.lower():
        model = GPTNeoXForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            cache_dir=cache_dir, 
            low_cpu_mem_usage=True, 
            device_map="auto"
        )
    elif 'gemma' in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            cache_dir=cache_dir, 
            low_cpu_mem_usage=True, 
            device_map="auto"
        )
    elif 'mistral' in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
    elif 'qwen' in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
    else:
        raise ValueError("Model not supported. Please use LLaMA, Pythia, Gemma, Mistral, or Qwen models.")

    model.seqlen = 1024
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=44, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')

    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument("--eval_zero_shot", action="store_true")
    parser.add_argument('--sparsity_schedule', type=str, default='uniform', choices=['uniform', 'linear', 'cosine', 'cubic', 'middle_high', 'middle_low', 'ramp_hold'], 
                        help='Per-layer sparsity schedule. uniform=same for all; '
                         'linear/cosine/cubic=ramp from sparsity_ratio_min to sparsity_ratio; '
                         'middle_high=ends low, middle high; middle_low=ends high, middle low.')
    parser.add_argument('--sparsity_ratio_min', type=float, default=0.0, help='Minimum sparsity for the earliest layers (used with non-uniform schedules).')
    parser.add_argument('--reverse_schedule', action='store_true', default=False, help='Reverse the schedule: layer 0 gets high sparsity, last layer gets low sparsity.')
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.eval()



    if 'llama' in args.model.lower():
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
        ctx = 4096
    elif 'pythia' in args.model.lower():
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        ctx = 2048
    elif 'gemma' in args.model.lower():
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        ctx = 4096
    elif 'mistral' in args.model.lower():
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        ctx = 4096
    else:
        raise ValueError("Model not supported. Please use LLaMA, Pythia, Gemma, or Mistral models.")
    
    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif "ablate" in args.prune_method:
            prune_ablate(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    ################################################################
    print("*"*30)
    sparsity_ratio, layer_sparsities = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################
    # ppl_test = eval_ppl(args, model, tokenizer, device)
    # print(f"wikitext perplexity {ppl_test}")

    ppl_test = evaluate_ppl('wikitext', model, tokenizer, ctx)
    print(f"wikitext perplexity {ppl_test}")

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")
    with open(save_filepath, "w") as f:
        print("method\tactual_sparsity\tppl_test", file=f, flush=True)
        print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}", file=f, flush=True)

    layer_sparsity_filepath = os.path.join(args.save, f"layer_sparsity_{args.prune_method}.txt")
    with open(layer_sparsity_filepath, "w") as f:
        print("layer\tsparsity", file=f, flush=True)
        for i, s in enumerate(layer_sparsities):
            print(f"{i}\t{s:.6f}", file=f, flush=True)

    if args.eval_zero_shot:
        accelerate=False
        if "30b" in args.model or "65b" in args.model or "70b" in args.model:
            accelerate=True

        task_list = ["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"]
        num_shot = 0
        results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, accelerate)
        print("********************************")
        print("zero_shot evaluation results")
        print(results)

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

if __name__ == '__main__':
    main()