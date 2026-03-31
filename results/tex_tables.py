import os
import re
import json
from collections import defaultdict

# ── config ────────────────────────────────────────────────────────────────────
methods         = ['dense', 'magnitude', 'wanda', 'sparsegpt']
model           = 'pythia_70m'
base_path       = '/research/nfs_khalili_17/gupte.31/reproduce/SAEBench'
sparsity_levels = [50]

METHOD_TEX = {
    'magnitude': r'\spm{Magnitude}',
    'wanda':     r'\spm{Wanda}',
    'sparsegpt': r'\spm{SparseGPT}',
}

# ── eval sources ──────────────────────────────────────────────────────────────
EVAL_SOURCES = [
    (
        'core',
        {
            'model_behavior_preservation':    ['kl_div_score'],
            'model_performance_preservation': ['ce_loss_score'],
            'reconstruction_quality':         ['explained_variance', 'cossim', 'mse'],
        }
    ),
    (
        'absorption',
        {
            'mean': ['mean_absorption_fraction_score', 'mean_full_absorption_score'],
        }
    ),
    (
        'scr/scr',
        {
            'scr_metrics': ['scr_metric_threshold_10'],
        }
    ),
    (
        'tpp/tpp',
        {
            'tpp_metrics': ['tpp_threshold_10_total_metric'],
        }
    ),
]

ALL_METRICS = [
    'kl_div_score',
    'ce_loss_score',
    'explained_variance',
    'cossim',
    'mse',
    'mean_absorption_fraction_score',
    'mean_full_absorption_score',
    'scr_metric_threshold_10',
    'tpp_threshold_10_total_metric',
]

# ── data collection ───────────────────────────────────────────────────────────
# data_store[filename][method][sp_key] = {metric: value, ...}
data_store    = defaultdict(lambda: defaultdict(dict))
all_filenames = set()

for eval_subdir, keys_to_collect in EVAL_SOURCES:
    for sp in sparsity_levels:
        for method in methods:
            if method == 'dense':
                directory = (
                    f'{base_path}/{model}/canonical/dense'
                    f'/eval_results/{eval_subdir}'
                )
                sp_key = 'dense'
            else:
                directory = (
                    f'{base_path}/{model}/canonical/sparsity={sp}'
                    f'/{method}/eval_results/{eval_subdir}'
                )
                sp_key = sp

            if not os.path.exists(directory):
                print(f'[WARNING] Directory not found, skipping: {directory}')
                continue

            for filename in sorted(
                f for f in os.listdir(directory) if '-res-' in f
            ):
                all_filenames.add(filename)
                filepath = os.path.join(directory, filename)
                if not os.path.isfile(filepath):
                    continue

                with open(filepath) as fh:
                    raw = json.load(fh)

                flat = {}
                for cat, metrics in keys_to_collect.items():
                    for metric in metrics:
                        flat[metric] = (
                            raw['eval_result_metrics'][cat].get(metric)
                        )

                existing = data_store[filename][method].get(sp_key, {})
                existing.update(flat)
                data_store[filename][method][sp_key] = existing

# ── helpers ───────────────────────────────────────────────────────────────────
def get_layer(filename):
    """Extract layer number from filenames across all supported model formats.
 
    Supported patterns:
      gemma_2_2b  : ..._layer_8_...
      gemma_2_9b  : ..._layer_8_...
      mistral-7b  : ...-res-wg_blocks.8.hook_...
      pythia-70m  : ...-res-sm_blocks.1.hook_...
    """
    # gemma: _layer_<n>_
    m = re.search(r'_layer_(\d+)_', filename)
    if m:
        return int(m.group(1))
    # mistral / pythia: blocks.<n>.hook
    m = re.search(r'blocks\.(\d+)\.hook', filename)
    if m:
        return int(m.group(1))
    return None


def group_avg(layer_ids, method, sp_key):
    """Average every metric over all files whose layer is in layer_ids."""
    layer_set = set(layer_ids)
    totals    = defaultdict(list)
    for fn in all_filenames:
        if get_layer(fn) not in layer_set:
            continue
        entry = data_store.get(fn, {}).get(method, {}).get(sp_key, {})
        for metric in ALL_METRICS:
            v = entry.get(metric)
            if v is not None:
                totals[metric].append(v)
    return {
        m: (sum(vs) / len(vs)) if vs else None
        for m, vs in totals.items()
    }


# Metrics where higher is better; absorption metrics are lower-is-better.
HIGHER_IS_BETTER = {
    'kl_div_score',
    'ce_loss_score',
    'explained_variance',
    'cossim',
    'scr_metric_threshold_10',
    'tpp_threshold_10_total_metric',
}


def f(v, bold=False):
    """Format a float (optionally bold) or return LaTeX placeholder."""
    if v is None:
        return r'\_'
    s = f'{v:.4f}'
    return f'\\textbf{{{s}}}' if bold else s


# ── shared table utilities ────────────────────────────────────────────────────
HEADER_LINES = [
    r'\begin{table*}[t]',
    r'\centering',
    r'\setlength{\tabcolsep}{3.5pt}',
    r'\footnotesize',
    r'\begin{tabular}{@{}llr ccccc cc c c@{}}',
    r'\toprule',
    (
        r'\multirow{2}{*}{\textbf{Layers}} &'
        r'\multirow{2}{*}{\textbf{Method}} &'
        r'\multirow{2}{*}{\makecell{\textbf{Spar.}\\\textbf{(\%)}}} &'
        r'\multicolumn{5}{c}{\textbf{Core}} &'
        r'\multicolumn{2}{c}{\textbf{Absorption}} &'
        r'\textbf{SCR} &'
        r'\textbf{TPP} \\'
    ),
    (
        r'\cmidrule(lr){4-8}\cmidrule(lr){9-10}'
        r'\cmidrule(lr){11-11}\cmidrule(lr){12-12}'
    ),
    (
        r'& & &'
        r'\makecell{KL\\Score$\uparrow$} &'
        r'\makecell{CE\\Score$\uparrow$} &'
        r'\makecell{Expl.\\Var.$\uparrow$} &'
        r'\makecell{Cos\\Sim$\uparrow$} &'
        r'\makecell{MSE\\$\downarrow$} &'
        r'\makecell{Abs.\\Frac.$\downarrow$} &'
        r'\makecell{Full\\Abs.$\downarrow$} &'
        r'\makecell{Top-10\\$\uparrow$} &'
        r'\makecell{Top-10\\$\uparrow$} \\'
    ),
    r'\midrule',
]

FOOTER_LINES = [r'\bottomrule', r'\end{tabular}', r'\end{table*}']


def make_row(method_tex, sp_tex, avgs, italic=False, best=None):
    """Render one data row (without the leading layer-label cell).

    best: dict mapping metric -> best value among non-baseline rows for this
          sparsity level.  When a cell matches its best value it is bolded.
          Pass None (default) to disable bolding (e.g. for the baseline row).
    """
    best = best or {}

    def cell(metric):
        v = avgs.get(metric)
        if v is None:
            return r'\_'
        bv = best.get(metric)
        is_best = (bv is not None) and (abs(v - bv) < 1e-9)
        return f(v, bold=is_best)

    kl   = cell('kl_div_score')
    ce   = cell('ce_loss_score')
    ev   = cell('explained_variance')
    cs   = cell('cossim')
    mse  = cell('mse')
    abf  = cell('mean_absorption_fraction_score')
    fabs = cell('mean_full_absorption_score')
    scr  = cell('scr_metric_threshold_10')
    tpp  = cell('tpp_threshold_10_total_metric')

    if italic:
        method_tex = f'\\textit{{{method_tex}}}'
        sp_tex     = f'\\textit{{{sp_tex}}}'
    return (
        f' & {method_tex} & {sp_tex}\n'
        f'   & {kl} & {ce} & {ev} & {cs} & {mse}\n'
        f'   & {abf} & {fabs}\n'
        f'   & {scr} & {tpp} \\\\'
    )


def compute_best(all_avgs):
    """Given a list of avg-dicts (one per method), return best value per metric.

    Higher-is-better metrics: best = max; lower-is-better: best = min.
    """
    best = {}
    for metric in ALL_METRICS:
        vals = [a[metric] for a in all_avgs if a.get(metric) is not None]
        if not vals:
            continue
        best[metric] = max(vals) if metric in HIGHER_IS_BETTER else min(vals)
    return best


def sparsity_block(layer_ids, lines):
    """
    Append baseline + all three sparsity-level blocks for a given set of
    layer_ids into `lines`.  Used by all three table builders.
    """
    sparse_methods = ['magnitude', 'wanda', 'sparsegpt']

    # baseline (dense) — no bolding
    avgs = group_avg(layer_ids, 'dense', 'dense')
    lines.append(make_row('Baseline', '0', avgs, italic=True, best=None))
    lines.append(r'\cmidrule(l){2-12}')

    for sp in sparsity_levels:
        # collect all method averages for this sparsity level first
        sp_avgs = {m: group_avg(layer_ids, m, sp) for m in sparse_methods}
        best    = compute_best(list(sp_avgs.values()))

        for method in sparse_methods:
            lines.append(make_row(METHOD_TEX[method], str(sp), sp_avgs[method], best=best))
        if sp != sparsity_levels[-1]:
            lines.append(r'\cmidrule(l){2-12}')


def build_table_per_layer_for_group(group_name, layer_ids):
    """One row-block per individual layer, restricted to a single group."""
    layers = sorted({get_layer(fn) for fn in all_filenames} - {None})
    layers = [l for l in layers if l in set(layer_ids)]

    # Derive a short slug for the label, e.g. "Early (0--8)" -> "early"
    slug = group_name.split()[0].lower()

    lines = HEADER_LINES[:]
    lines[2] = rf'\label{{tab:results_per_layer_{slug}}}'

    for l_idx, layer in enumerate(layers):
        n_rows = 1 + len(sparsity_levels) * 3
        lines.append(
            f'\\multirow{{{n_rows}}}{{*}}'
            f'{{\\rotatebox[origin=c]{{90}}{{\\texttt{{Layer {layer}}}}}}}'
        )
        sparsity_block([layer], lines)
        if l_idx < len(layers) - 1:
            lines.append(r'\midrule')

    lines += FOOTER_LINES
    return '\n'.join(lines)



# ── Table 3 : global average (all layers) ────────────────────────────────────
def build_table_global_avg():
    """Single row-block averaged across every layer."""
    all_layers = sorted({get_layer(fn) for fn in all_filenames} - {None})

    lines = HEADER_LINES[:]
    lines[2] = r'\label{tab:results_global_avg}'

    n_rows = 1 + len(sparsity_levels) * 3
    lines.append(
        f'\\multirow{{{n_rows}}}{{*}}'
        f'{{\\rotatebox[origin=c]{{90}}{{\\texttt{{All Layers}}}}}}'
    )
    sparsity_block(all_layers, lines)

    lines += FOOTER_LINES
    return '\n'.join(lines)

# ── Table 4 : all layers, no grouping ────────────────────────────────────────
def build_table_all_layers():
    """One row-block per individual layer across all layers, no group separation."""
    all_layers = sorted({get_layer(fn) for fn in all_filenames} - {None})
 
    lines = HEADER_LINES[:]
    lines[2] = r'\label{tab:results_all_layers}'
 
    for l_idx, layer in enumerate(all_layers):
        n_rows = 1 + len(sparsity_levels) * 3
        lines.append(
            f'\\multirow{{{n_rows}}}{{*}}'
            f'{{\\rotatebox[origin=c]{{90}}{{\\texttt{{Layer {layer}}}}}}}'
        )
        sparsity_block([layer], lines)
        if l_idx < len(all_layers) - 1:
            lines.append(r'\midrule')
 
    lines += FOOTER_LINES
    return '\n'.join(lines)
 

# ── optional per-file debug print ────────────────────────────────────────────
def print_per_file():
    for filename in sorted(all_filenames):
        print(f"\n{'='*70}\nFILE: {filename}\n{'='*70}")
        for method in methods:
            if method not in data_store[filename]:
                continue
            sp_keys = ['dense'] if method == 'dense' else sparsity_levels
            for sp_key in sp_keys:
                d = data_store[filename][method].get(sp_key, {})
                label = 'dense' if sp_key == 'dense' else f'{method}_sp{sp_key}'
                for metric in ALL_METRICS:
                    v = d.get(metric)
                    if v is not None:
                        print(f'  {label:<22} / {metric}: {v:.4f}')


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Uncomment to see per-file debug output:
    # print_per_file()

    out_dir = f'{model}/tables'
    os.makedirs(out_dir, exist_ok=True)

    tables = {
        'latex_table_global_avg.tex': build_table_global_avg(),
        'latex_table_all_layers.tex':  build_table_all_layers(),
    }

    for filename, latex in tables.items():
        out_path = os.path.join(out_dir, filename)
        with open(out_path, 'w') as fh:
            fh.write(latex)
        print(f'[INFO] Written: {out_path}')

