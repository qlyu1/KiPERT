"""Twelve complementary metrics; within-dataset ranks averaged equally over five datasets."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

METHODS = ['UC', 'biolord', 'scouter', 'txpert', 'gears', 'stamp', 'regformer']
NAMES = ['UC-WFR', 'biolord', 'Scouter', 'TxPert-public', 'GEARS', 'STAMP–Geneformer', 'RegFormer–GEARS']
DATASETS = ['S3', 'S4', 'S5', 'S6', 'S7']
COLORS = ['#174F64', '#D58B55', '#569B80', '#AC82A8', '#648AC0', '#AD9A4E', '#70777F']
METRICS = [
    ('mse', 'All-gene\nMSE', '全基因 MSE', True),
    ('pearson_delta', 'All-gene\nPearson Δ', '全基因 Pearson Δ', False),
    ('systema_cell_pearson', 'Systema\nPearson', 'Systema 校正 Pearson', False),
    ('sp_mse100', 'Top-100\nMSE', 'Top-100 MSE', True),
    ('sp_pcc100', 'Top-100\nPCC Δ', 'Top-100 PCC Δ', False),
    ('auprc_dg', 'DEG\nAUPRC', 'DEG AUPRC（非线性插值）', False),
    ('deg_precision100', 'DEG\nprecision@100', 'DEG precision@100', False),
    ('deg_recall100', 'DEG\nrecall@100', 'DEG recall@100', False),
    ('top100_direction', 'Top-100\ndirection', 'Top-100方向一致率', False),
    ('pds_query_l2', 'L2\nPDS', 'L2 PDS', False),
    ('pds_query_cosine', 'Cosine\nPDS', 'Cosine PDS', False),
    ('pb_reverse_rank_l2', 'Reverse L2\nretrieval', '反向L2扰动检索名次', True),
]

def render_radar(output, summary):
    output = Path(output)
    figures = output / 'figures'
    records = []
    assert len(METRICS) == len({x[0] for x in METRICS}) == 12
    for dataset in DATASETS:
        scores = summary[summary.dataset == dataset].set_index('method').loc[METHODS]
        for metric, _, cn, lower in METRICS:
            assert np.isfinite(scores[metric]).all()
            ranks = scores[metric].rank(ascending=lower, method='average')
            for method in METHODS:
                records.append(dict(dataset=dataset, method=method, metric=metric,
                                    metric_label=cn, direction='lower' if lower else 'higher',
                                    mean_value=scores.loc[method, metric], rank=ranks[method],
                                    radar_score=(7-ranks[method])/6))
    detail = pd.DataFrame(records)
    detail.to_csv(output/'雷达图_原始分数与归一化排名.csv', index=False, encoding='utf-8-sig')
    means = detail.groupby(['method', 'metric'], sort=False).agg(
        mean_rank=('rank', 'mean'), mean_score=('radar_score', 'mean'),
        dataset_count=('dataset', 'nunique')).reset_index()
    assert len(detail) == 420 and len(means) == 84 and (means.dataset_count == 5).all()
    np.testing.assert_allclose(means.mean_score, (7-means.mean_rank)/6)
    means['metric_label'] = means.metric.map({m: cn for m, _, cn, _ in METRICS})
    means.to_csv(output/'雷达图_多指标五数据集平均排名.csv', index=False, encoding='utf-8-sig')
    means.pivot(index='metric_label', columns='method', values='mean_rank').reindex(
        index=[x[2] for x in METRICS], columns=METHODS).rename(columns=dict(zip(METHODS, NAMES))).to_csv(
        output/'雷达图_平均排名对照表.csv', encoding='utf-8-sig')
    definition = dict(metrics=[dict(key=m, label=cn, lower_better=lo) for m, _, cn, lo in METRICS],
        dataset_ids=DATASETS, dataset_weight=0.2, methods=METHODS, rank_ties='average',
        formula='rbar(m,k)=sum_d rank_d(m,k)/5; radius=(7-rbar)/6',
        source='source_data/summary.csv; averages of unique-target scores',
        pds='Query target gene excluded; original raw-response PDS; no oracle rescaling',
        systema='Cell-count-weighted reference as implemented in the paper-matched benchmark',
        auprc='Nonlinear interpolation, not the earlier average-precision metric',
        aggregation='Equal dataset weights; point estimates; no composite score or polygon-area statistic',
        reverse_retrieval='L2 reverse normalized rank: fixed truth query ranked among candidate model predictions; lower is better',
        direction='Sign agreement on 100 truth genes with highest absolute Welch t statistic; no significance filter',
        deg_shortlist='Precision and recall of true DEG labels among 100 genes with largest predicted absolute changes')
    (output/'RADAR_DEFINITION.json').write_text(json.dumps(definition, ensure_ascii=False, indent=2))

    previous_style = plt.rcParams.copy()
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['DejaVu Sans','Arial'],
        'font.size':7, 'xtick.labelsize':7, 'ytick.labelsize':6, 'legend.fontsize':6.4,
        'pdf.fonttype':42, 'svg.fonttype':'none', 'legend.frameon':False, 'savefig.facecolor':'white'})
    fig = plt.figure(figsize=(183/25.4, 190/25.4))
    fig.text(.055, .98, 'Unseen perturbation prediction | Mean ranks', fontsize=9, va='top')
    fig.text(.055, .953, '12 complementary metrics · Five equally weighted datasets · Outer is better', fontsize=6.7, va='top', color='#4C5965')
    ax = fig.add_axes([.20,.23,.60,.60], projection='polar')
    angles = np.linspace(0, 2*np.pi, len(METRICS), endpoint=False)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    for method, color in zip(METHODS, COLORS):
        vals = means[means.method == method].set_index('metric').loc[[x[0] for x in METRICS], 'mean_score'].to_numpy()
        ax.plot(np.r_[angles, angles[0]], np.r_[vals, vals[0]], 'o-', ms=2.8, color=color,
                lw=1.3 if method == 'UC' else .85, zorder=4 if method == 'UC' else 3)
    ax.set_ylim(0,1)
    ax.set_xticks(angles, [x[1] for x in METRICS])
    ax.tick_params(axis='x', pad=11)
    ax.set_yticks([0,1/3,2/3,1], ['', '', '', ''])
    ax.grid(color='#CBD2D6', lw=.5)
    ax.spines['polar'].set_color('#CBD2D6')
    fig.text(.5,.155, 'Mean rank, centre → outer rings: 7 → 5 → 3 → 1', ha='center', fontsize=6.5)
    fig.legend(handles=[Line2D([0],[0],color=c,marker='o',ms=2.8,lw=1,label=n)
        for c,n in zip(COLORS,NAMES)], loc='lower center', bbox_to_anchor=(.5,.07), ncol=4,
        columnspacing=1.2, handlelength=1.3)
    fig.text(.055,.034, 'Each axis averages within-dataset ranks over Tian CRISPRi, Tian CRISPRa, Adamson, Norman singles and Frangieh.', fontsize=5.7)
    fig.text(.055,.018, 'Radius = (7 − mean rank)/6. Average ties; MSE and reverse rank: lower is better; other metrics: higher.', fontsize=5.7)
    fig.canvas.draw()
    (figures/'03_Top100_radar.alignment.json').write_text(json.dumps(dict(
        verdict='NOT APPLICABLE', reason='One polar plot; no comparable panel alignment',
        n_panels=1, tolerance_pt=1.5), indent=2))
    fig.savefig(figures/'03_Top100_radar.pdf')
    fig.savefig(figures/'03_Top100_radar.svg')
    fig.savefig(figures/'03_Top100_radar.png', dpi=600)
    fig.savefig(figures/'03_Top100_radar.preview.png', dpi=300)
    plt.close(fig)
    plt.rcParams.update(previous_style)
    print('RADAR_READY: 12 complementary metrics; 420 dataset values; 84 five-dataset means', flush=True)

if __name__ == '__main__':
    import sys
    output = Path(sys.argv[1])
    render_radar(output, pd.read_csv(output/'source_data/summary.csv'))
