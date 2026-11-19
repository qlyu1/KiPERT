from pathlib import Path
import json
B = Path(__file__).resolve().parents[1]
P = B.parent / 'pie_four_context_reproduction_20261106'
FOLDS = ['hepg2', 'jurkat', 'k562', 'rpe1']
MODEL = dict(n_genes=6642, n_hidden=100, token_dim=16, n_cells=128,
             n_layers=9, n_heads=8, mlp_ratio=4, dropout=0.,
             mask_rate_min=.1, mask_rate_max=.8, sw_weight=.01, n_proj=64)
GENERATION = dict(num_steps=5, prompt_ratio=.25, context_ratio=.4,
                  context_ratio_min=.2, mask_rate=1., mode='mdm',
                  gene_name_col='gene_name', batch_size=2,
                  show_progress=False, num_workers=0)
PROTOCOL = {
    'repository': 'https://github.com/ArcInstitute/stack',
    'commit': 'cacc2e4b09435c3e536d46237d10b50f222dd144',
    'paper_specified': {'split': 'PIE unseen_ctx four folds', 'genes': 6642,
        'raw_counts': 'same 611211 filtered cells and gene order as existing benchmark',
        'pretrain_peak_lr': 3e-5, 'cells_per_set': 128,
        'stages': ['from-scratch masked-gene pretraining', 'in-context posttraining'],
        'selection': 'minimum validation loss in each stage',
        'prompt': 'same-perturbation cells from allowed training contexts',
        'query': '500 held-context controls', 'DE_reference': 'STACK-generated controls'},
    'unpublished_settings_filled_from_official_presets': {
        'architecture': 'configs/training/bc_large.yaml and training CLI defaults',
        'pretrain_epochs': 10, 'posttrain_epochs': 8,
        'posttrain': 'configs/finetuning/ft_parsecg.yaml',
        'posttrain_lr': 2e-5, 'weight_decay': .003, 'replacement_ratio': .75,
        'generation': GENERATION},
    'model_config': MODEL,
    'execution': {'seed': 42, 'devices': 2, 'microbatch_per_device': 2,
        'gradient_accumulation': 8, 'effective_batch_size': 32,
        'precision': 'bf16-mixed', 'gradient_clip': 1.,
        'order': 'one fold at a time on both GPUs after all STATE scoring',
        'checkpoint_retention': 'best posttraining weights retained; pretraining intermediate removed only after successful posttraining; current full resume checkpoint in RAM'},
    'limits': ['PIE does not release the complete STACK training configuration; this is a local implementation of the described protocol, not a claim of exact author-run replication.',
        'PIE does not specify small-group sampling. Our adapter retains all PIE conditions by padding groups below 128 cells with same-condition resampling, using 32 prompt cells and 96 matched-context control queries. Official complete blocks and losses are unchanged; all padded groups are recorded.'],
    'evaluation': {'gpudge': '0.9.1', 'normalize_target_sum': 10000,
        'filter_gene_min_cpm_cell': 5, 'query_cells': 500,
        'scorer': 'unmodified PIE scorer', 'q_to_pde': 's/(s+s0), s=-log10(max(q,1e-10)), s0=-log10(.05)',
        'filtered_genes': 'q=1, lfc=0', 'delta': 'mean log1p(CP10k predicted cells) minus mean log1p(CP10k generated controls)'}
}

def write_protocol():
    (B / 'manifests/stack_protocol.json').write_text(json.dumps(PROTOCOL, indent=2))

if __name__ == '__main__':
    write_protocol()
