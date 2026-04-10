#!/usr/bin/env python3
"""Generate LaTeX snippet files from results JSON.

v2: Updated for config-level means, fold std, coverage matrix,
    per-model LOMO, analytical in-sample R².
    Uses log-space k-fold as primary within-platform result.
"""
import json, os

RESULTS = os.path.expanduser('~/Documents/project_costmodel_tcas2/results/all_results.json')
TEX_DIR = os.path.expanduser('~/Documents/project_costmodel_tcas2/results/tex/')
os.makedirs(TEX_DIR, exist_ok=True)

with open(RESULTS) as f:
    R = json.load(f)

def w(name, val):
    with open(os.path.join(TEX_DIR, name), 'w') as f:
        f.write(str(val))

# Dataset stats
w('n_raw_records.tex', f"{R['dataset']['n_raw_records']:,}")
w('n_configs.tex', str(R['dataset']['n_configs']))
w('n_records.tex', str(R['dataset']['n_configs']))
w('n_models.tex', str(R['dataset']['n_models']))
w('n_modalities.tex', str(R['dataset']['n_modalities']))
w('mean_reps.tex', f"{R['dataset']['mean_reps_per_config']:.1f}")

# 5-fold: use log-space k-fold as primary (trained in log-space, evaluated in raw space)
kf = R['kfold_energy_logspace']
kf_folds = R.get('kfold_energy_logspace_folds', {})
best_kf = max(kf.items(), key=lambda x: x[1]['R2'])
best_kf_name = best_kf[0]
w('kfold_best_r2.tex', f"{best_kf[1]['R2']:.3f}")
w('kfold_best_mape.tex', f"{best_kf[1]['MAPE']:.1f}")
w('kfold_best_method.tex', best_kf_name)
# Fold std
if best_kf_name in kf_folds:
    w('kfold_best_r2_std.tex', f"{kf_folds[best_kf_name]['R2_std']:.3f}")
    w('kfold_best_mape_std.tex', f"{kf_folds[best_kf_name]['MAPE_std']:.1f}")

# Also save raw-feature kfold for comparison
kf_raw = R.get('kfold_energy_raw', {})
if kf_raw:
    best_kf_raw = max(kf_raw.items(), key=lambda x: x[1]['R2'])
    w('kfold_raw_best_r2.tex', f"{best_kf_raw[1]['R2']:.3f}")
    w('kfold_raw_best_mape.tex', f"{best_kf_raw[1]['MAPE']:.1f}")

# LOHO XGBoost
loho = R['loho_energy_raw']
w('loho_xgb_r2.tex', f"{loho['XGBoost']['R2']:.2f}")
loho_best = max(loho.items(), key=lambda x: x[1]['R2'])
w('loho_best_r2.tex', f"{loho_best[1]['R2']:.2f}")
w('loho_best_mape.tex', f"{loho_best[1]['MAPE']:.0f}")

# LOHO per-hw A100
loho_hw = R['loho_energy_raw_per_hw']
a100_mapes = [(m, d['A100']['MAPE']) for m, d in loho_hw.items() if 'A100' in d]
best_a100 = min(a100_mapes, key=lambda x: x[1])
w('loho_a100_mape.tex', f"{best_a100[1]:.1f}")

# LOMO best
lomo = R['lomo_energy']
best_lomo = max(lomo.items(), key=lambda x: x[1]['R2'])
w('lomo_best_r2.tex', f"{best_lomo[1]['R2']:.2f}")
w('lomo_best_method.tex', best_lomo[0])

# Analytical
anal = R['analytical']
w('analytical_loho_mape.tex', f"{anal['loho_energy_overall']['MAPE']:.0f}")
w('analytical_insample_r2.tex', f"{anal['insample_energy_r2']:.2f}")

# SHAP values
shap_ar = R['shap_per_arch']['AR']
shap_diff = R['shap_per_arch']['Diffusion']
w('shap_ar_tdp.tex', f"{shap_ar['log_tdp']:.2f}")
w('shap_ar_complex.tex', f"{shap_ar['log_complexity']:.2f}")
w('shap_diff_complex.tex', f"{shap_diff['log_complexity']:.2f}")
w('shap_diff_steps.tex', f"{shap_diff['log_steps']:.2f}")
w('shap_diff_tdp.tex', f"{shap_diff['log_tdp']:.2f}")

# ---- TABLES ----

models_order = ['Linear', 'Ridge', 'RF', 'XGBoost', 'SVR']

# Table: 5-fold CV (log-space training, raw-space metrics — primary result)
lines = []
lines.append(r'\begin{tabular}{@{}lrrr@{}}')
lines.append(r'\toprule')
lines.append(r'\textbf{Method} & \textbf{R\textsuperscript{2}} & \textbf{RMSE (J)} & \textbf{MAPE (\%)} \\')
lines.append(r'\midrule')
for m in models_order:
    d = kf[m]
    bold = r'\textbf' if d['R2'] == best_kf[1]['R2'] else ''
    r2s = f"{bold}{{{d['R2']:.3f}}}" if bold else f"{d['R2']:.3f}"
    lines.append(f"{m} & {r2s} & {d['RMSE']:.1f} & {d['MAPE']:.1f} \\\\")
lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')
w('kfold_table.tex', '\n'.join(lines))

# Table: LOHO CV
lines = []
lines.append(r'\begin{tabular}{@{}lrrr@{}}')
lines.append(r'\toprule')
lines.append(r'\textbf{Method} & \textbf{R\textsuperscript{2}} & \textbf{RMSE (J)} & \textbf{MAPE (\%)} \\')
lines.append(r'\midrule')
for m in models_order:
    d = loho[m]
    lines.append(f"{m} & {d['R2']:.2f} & {d['RMSE']:.1f} & {d['MAPE']:.1f} \\\\")
lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')
w('loho_table.tex', '\n'.join(lines))

# Table: LOHO per-hardware
hw_names = ['M4 Pro', 'A100', 'H100']
lines = []
lines.append(r'\begin{tabular}{@{}l' + 'r' * len(hw_names) + r'@{}}')
lines.append(r'\toprule')
lines.append(r'\textbf{Method} & ' + ' & '.join([f'\\textbf{{{h}}}' for h in hw_names]) + r' \\')
lines.append(r'\midrule')
for m in models_order:
    vals = []
    for h in hw_names:
        if h in loho_hw[m]:
            vals.append(f"{loho_hw[m][h]['MAPE']:.1f}")
        else:
            vals.append('--')
    lines.append(f"{m} & " + " & ".join(vals) + r" \\")
lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')
w('loho_hw_table.tex', '\n'.join(lines))

# Table: LOMO CV
lines = []
lines.append(r'\begin{tabular}{@{}lrrr@{}}')
lines.append(r'\toprule')
lines.append(r'\textbf{Method} & \textbf{R\textsuperscript{2}} & \textbf{RMSE (J)} & \textbf{MAPE (\%)} \\')
lines.append(r'\midrule')
for m in models_order:
    d = lomo[m]
    lines.append(f"{m} & {d['R2']:.2f} & {d['RMSE']:.1f} & {d['MAPE']:.1f} \\\\")
lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')
w('lomo_table.tex', '\n'.join(lines))

# Coverage matrix table
coverage = R.get('coverage_matrix', {})
if coverage:
    lines = []
    lines.append(r'\begin{tabular}{@{}lccc@{}}')
    lines.append(r'\toprule')
    lines.append(r'\textbf{Model} & \textbf{M4 Pro} & \textbf{A100} & \textbf{H100} \\')
    lines.append(r'\midrule')
    for model in sorted(coverage.keys()):
        hw = coverage[model]
        cells = []
        for h in ['M4 Pro', 'A100', 'H100']:
            n = hw.get(h, 0)
            cells.append(r'\checkmark' if n > 0 else '--')
        lines.append(f"{model} & " + " & ".join(cells) + r" \\")
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    w('coverage_table.tex', '\n'.join(lines))

print(f"Generated {len(os.listdir(TEX_DIR))} TeX snippet files in {TEX_DIR}")
