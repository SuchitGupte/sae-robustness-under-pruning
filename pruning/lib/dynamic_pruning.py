import copy
import math
import time
import heapq
import torch
import random
import torch.nn as nn
from .sparsegpt import SparseGPT
from .layerwrapper import WrappedGPT
from .data import get_loaders
import pdb
import gc
from .ablate import AblateGPT


def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def get_model_layers(model):
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'gpt_neox'):
        return model.gpt_neox.layers
    else:
        raise ValueError(f"Unsupported model architecture: {type(model)}")

def get_embed_token_key(model):
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return "model.embed_tokens"
    elif hasattr(model, 'gpt_neox'):
        return "gpt_neox.embed_in"
    return "model.embed_tokens"

def get_layer_key(model, i):
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return f"model.layers.{i}"
    elif hasattr(model, 'gpt_neox'):
        return f"gpt_neox.layers.{i}"
    return f"model.layers.{i}"

def forward_layer(layer, inp, attention_mask, position_ids, model):
    if hasattr(model, 'gpt_neox'):
        seq_len = inp.shape[1]
        pos = position_ids if position_ids is not None else torch.arange(seq_len, device=inp.device).unsqueeze(0)
        position_embeddings = layer.attention.rotary_emb(inp, pos)
        return layer(inp, attention_mask=attention_mask, position_embeddings=position_embeddings)[0]
    elif 'gemma' in type(model).__name__.lower():
        seq_len = inp.shape[1]
        pos = position_ids if position_ids is not None else torch.arange(seq_len, device=inp.device).unsqueeze(0)
        position_embeddings = model.model.rotary_emb(inp, pos)
        return layer(inp, attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
    elif 'mistral' in type(model).__name__.lower():
        seq_len = inp.shape[1]
        pos = position_ids if position_ids is not None else torch.arange(seq_len, device=inp.device).unsqueeze(0)
        position_embeddings = model.model.rotary_emb(inp, pos)
        return layer(inp, attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
    elif 'qwen' in type(model).__name__.lower():
        seq_len = inp.shape[1]
        pos = position_ids if position_ids is not None else torch.arange(seq_len, device=inp.device).unsqueeze(0)
        position_embeddings = model.model.rotary_emb(inp, pos)
        return layer(inp, attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
    else:
        return layer(inp, attention_mask=attention_mask, position_ids=position_ids)[0]

def check_sparsity(model):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = get_model_layers(model)
    count = 0
    total_params = 0
    layer_sparsities = []
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        layer_sparsity = float(sub_count) / sub_params
        layer_sparsities.append(layer_sparsity)
        print(f"layer {i} sparsity {layer_sparsity:.6f}")

    model.config.use_cache = use_cache
    return float(count) / total_params, layer_sparsities

def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = get_model_layers(model)

    embed_key = get_embed_token_key(model)
    if embed_key in model.hf_device_map:
        device = model.hf_device_map[embed_key]

    dtype = next(iter(model.parameters())).dtype
    inps = []
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps.append(inp.detach().to('cpu'))
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs.get('position_ids', None)
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = []
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids.to('cpu') if position_ids is not None else None

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def get_layer_sparsity(layer_idx, num_layers, args):
    """
    Compute per-layer sparsity ratio.

    Schedules:
      'uniform'     : all layers use args.sparsity_ratio (default)
      'linear'      : ramps from sparsity_ratio_min -> sparsity_ratio
      'cosine'      : cosine ramp from sparsity_ratio_min -> sparsity_ratio
      'cubic'       : cubic ramp from sparsity_ratio_min -> sparsity_ratio
      'middle_high' : middle layers = sparsity_ratio (high), both ends = sparsity_ratio_min (low)
      'middle_low'  : middle layers = sparsity_ratio_min (low), both ends = sparsity_ratio (high)

    If --reverse_schedule is set, the ramp direction is flipped for linear/cosine/cubic.
    (Has no effect on middle_high / middle_low which are already symmetric.)
    """
    schedule = getattr(args, 'sparsity_schedule', 'uniform')
    if schedule == 'uniform' or num_layers == 1:
        return args.sparsity_ratio

    min_s = getattr(args, 'sparsity_ratio_min', 0.0)
    max_s = args.sparsity_ratio
    t = layer_idx / (num_layers - 1)  # 0.0 at first layer, 1.0 at last

    # ramp from min_s -> max_s in first half, hold at max_s for second half
    # average sparsity = 0.5 when: min_s + 3*max_s = 2.0  (e.g. 0.20+0.60, 0.35+0.55)
    if schedule == 'ramp_hold':
        if t <= 0.5:
            f = t / 0.5  # linearly 0->1 over first half
        else:
            f = 1.0      # hold at max for second half
        return min_s + (max_s - min_s) * f

    # symmetric schedules — reverse_schedule has no meaning here
    if schedule == 'middle_high':
        # peaks at center: f(t) = 1 - 2*|t - 0.5|  (triangle, 0 at ends, 1 at middle)
        f = 1.0 - 2.0 * abs(t - 0.5)
        return min_s + (max_s - min_s) * f
    elif schedule == 'middle_low':
        # valley at center: f(t) = 2*|t - 0.5|  (1 at ends, 0 at middle)
        f = 2.0 * abs(t - 0.5)
        return min_s + (max_s - min_s) * f

    if getattr(args, 'reverse_schedule', False):
        t = 1.0 - t  # flip: first layer gets max, last gets min

    if schedule == 'linear':
        return min_s + (max_s - min_s) * t
    elif schedule == 'cosine':
        return min_s + (max_s - min_s) * (1 - math.cos(math.pi * t)) / 2
    elif schedule == 'cubic':
        return min_s + (max_s - min_s) * (t ** 3)
    else:
        raise ValueError(f"Unknown sparsity_schedule: '{schedule}'. "
                         f"Choose from uniform, linear, cosine, cubic, middle_high, middle_low.")


def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = get_model_layers(model)

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        layer_sparsity = get_layer_sparsity(i, len(layers), args)

        for name in subset:
            W = subset[name].weight.data
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*layer_sparsity)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0


def prune_wandaplus(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    print("loading calibration data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
    layers = get_model_layers(model).to('cpu')

    for i in range(len(layers)):
        print(f'Beginning pruning layer {i}')
        layer_sparsity = get_layer_sparsity(i, len(layers), args)

        layer_args = copy.deepcopy(args)
        layer_args.sparsity_ratio = layer_sparsity

        layer_wrapper_config = copy.deepcopy(model.config)
        layer_wrapper_config.args = layer_args
        layer_wrapper_config.prune_n = prune_n
        layer_wrapper_config.prune_m = prune_m

        old_layer = layers[i]
        layers[i] = WeightWandaPlusPruner.wrap(
            module=old_layer,
            wrapper_config=layer_wrapper_config,
            layer_idx=i,
        )
        layers[i].to(device)
        inps = layers[i].pruning(inps, attention_mask, position_ids)
        torch.cuda.empty_cache()


def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibration data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = get_model_layers(model)

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        layer_sparsity = get_layer_sparsity(i, len(layers), args)
        outs = [None] * len(inps)

        dev = model.hf_device_map.get(get_layer_key(model, i), device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(dev)
        position_ids = position_ids.to(dev) if position_ids is not None else None

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = forward_layer(layer, inps[j].to(dev), attention_mask=attention_mask, position_ids=position_ids, model=model)
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name} sparsity {layer_sparsity:.4f}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)).to(subset[name].weight.device))

            W_mask = (torch.zeros_like(W_metric) == 1)
            if prune_n != 0:
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - layer_sparsity) > 0.001) and (alpha_hist[1]-alpha_hist[0] >= 0.001):
                        if cur_sparsity > layer_sparsity:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    indices = sort_res[1][:, :int(W_metric.shape[1] * layer_sparsity)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = forward_layer(layer, inps[j].to(dev), attention_mask=attention_mask, position_ids=position_ids, model=model)
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = get_model_layers(model)

    embed_key = get_embed_token_key(model)
    if embed_key in model.hf_device_map:
        dev = model.hf_device_map[embed_key]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs.get('position_ids', None)
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        layer_sparsity = get_layer_sparsity(i, len(layers), args)

        layer_key = get_layer_key(model, i)
        if layer_key in model.hf_device_map:
            dev = model.hf_device_map[layer_key]
            print(f"layer {i} device {dev}")
            inps, outs = inps.to(dev), outs.to(dev)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dev)
            position_ids = position_ids.to(dev) if position_ids is not None else None

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = forward_layer(layer, inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, model=model).squeeze(0)
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')
            gpts[name].fasterprune(layer_sparsity, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = forward_layer(layer, inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, model=model).squeeze(0)

        layers[i] = layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = get_model_layers(model)

    embed_key = get_embed_token_key(model)
    if embed_key in model.hf_device_map:
        dev = model.hf_device_map[embed_key]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs.get('position_ids', None)
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        layer_sparsity = get_layer_sparsity(i, len(layers), args)

        layer_key = get_layer_key(model, i)
        if layer_key in model.hf_device_map:
            dev = model.hf_device_map[layer_key]
            print(f"layer {i} device {dev}")
            inps, outs = inps.to(dev), outs.to(dev)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dev)
            if position_ids is not None:
                position_ids = position_ids.to(dev)
        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = AblateGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = forward_layer(layer, inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, model=model).squeeze(0)
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            if args.prune_method == "ablate_wanda_seq":
                prune_mask = gpts[name].get_wanda_mask(layer_sparsity, prune_n, prune_m)
            elif args.prune_method == "ablate_mag_seq":
                prune_mask = gpts[name].get_mag_mask(layer_sparsity, prune_n, prune_m)
            elif "iter" in args.prune_method:
                prune_mask = None

            gpts[name].fasterprune(args, layer_sparsity, mask=prune_mask, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = forward_layer(layer, inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, model=model).squeeze(0)

        layers[i] = layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
