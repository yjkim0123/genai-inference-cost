#!/usr/bin/env python3
"""
Predictive Modeling of Generative AI Inference Cost Across Hardware Architectures
IEEE TCAS-II Express Briefs — Cost Model Training & Evaluation

Author: Yongjun Kim, Ajou University

v2: Fixed data leakage — aggregated to config-level means before CV.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTS
# ============================================================
HARDWARE_SPECS = {
    'Apple M4 Pro': {
        'tdp': 40, 'compute_units': 20, 'memory_bw': 273, 'hw_id': 0
    },
    'NVIDIA A100-SXM4-40GB': {
        'tdp': 400, 'compute_units': 6912, 'memory_bw': 1555, 'hw_id': 1
    },
    'NVIDIA H100 80GB HBM3': {
        'tdp': 700, 'compute_units': 14592, 'memory_bw': 3350, 'hw_id': 2
    }
}

MODEL_INFO = {
    'Phi-3-mini-4k':       {'arch': 'AR',   'params_B': 3.8},
    'Mistral-7B-Instruct': {'arch': 'AR',   'params_B': 7.2},
    'MusicGen-small':      {'arch': 'AR',   'params_B': 0.589},
    'MusicGen-medium':     {'arch': 'AR',   'params_B': 1.5},
    'SD-v1-5':             {'arch': 'Diff', 'params_B': 0.9},
    'SDXL-base-1.0':       {'arch': 'Diff', 'params_B': 3.5},
    'AnimateDiff-v1.5':    {'arch': 'Diff', 'params_B': 0.9},
}

DATA_DIR = os.path.expanduser('~/Documents/project_energy/results/')
OUT_DIR  = os.path.expanduser('~/Documents/project_costmodel_tcas2/results/')
FIG_DIR  = os.path.expanduser('~/Documents/project_costmodel_tcas2/figures/')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

HW_SHORT = {
    'Apple M4 Pro': 'M4 Pro',
    'NVIDIA A100-SXM4-40GB': 'A100',
    'NVIDIA H100 80GB HBM3': 'H100'
}

def parse_resolution(s):
    if not s or s == '?':
        return 0
    parts = s.split('x')
    return int(parts[0]) * int(parts[1])

def compute_output_complexity(rec, modality):
    if modality in ('text', 'text_batched'):
        return rec.get('max_tokens', rec.get('actual_tokens', 64))
    elif modality == 'image':
        return parse_resolution(rec.get('resolution', '256x256'))
    elif modality == 'video':
        px = parse_resolution(rec.get('resolution', '256x256'))
        return px * rec.get('frames', 4)
    elif modality == 'music':
        return rec.get('max_tokens', 128)
    return 1

# ============================================================
# DATA LOADING
# ============================================================
def load_all_data():
    file_configs = [
        ('exp1_text.json',              'Apple M4 Pro',            'text'),
        ('exp2_image.json',             'Apple M4 Pro',            'image'),
        ('exp3_video.json',             'Apple M4 Pro',            'video'),
        ('exp4_music.json',             'Apple M4 Pro',            'music'),
        ('exp2_image_extra_Mac.json',   'Apple M4 Pro',            'image'),
        ('exp3_video_extra_Mac.json',   'Apple M4 Pro',            'video'),
        ('exp5_sdxl_Mac.json',          'Apple M4 Pro',            'image'),
        ('exp6_mistral_Mac.json',       'Apple M4 Pro',            'text'),
        ('exp1_text_A100.json',         'NVIDIA A100-SXM4-40GB',   'text'),
        ('exp2_image_A100.json',        'NVIDIA A100-SXM4-40GB',   'image'),
        ('exp2_image_extra_A100.json',  'NVIDIA A100-SXM4-40GB',   'image'),
        ('exp3_video_A100.json',        'NVIDIA A100-SXM4-40GB',   'video'),
        ('exp3_video_extra_A100.json',  'NVIDIA A100-SXM4-40GB',   'video'),
        ('exp4_music_A100.json',        'NVIDIA A100-SXM4-40GB',   'music'),
        ('exp5_sdxl_A100.json',         'NVIDIA A100-SXM4-40GB',   'image'),
        ('exp6_mistral_A100.json',      'NVIDIA A100-SXM4-40GB',   'text'),
        ('text_phi3_H100.json',         'NVIDIA H100 80GB HBM3',   'text'),
        ('image_sd15_H100.json',        'NVIDIA H100 80GB HBM3',   'image'),
        ('image_sdxl_H100.json',        'NVIDIA H100 80GB HBM3',   'image'),
        ('video_animatediff_H100.json', 'NVIDIA H100 80GB HBM3',   'video'),
        ('exp7_batched_A100.json',      'NVIDIA A100-SXM4-40GB',   'text_batched'),
        ('batched_phi3_H100.json',      'NVIDIA H100 80GB HBM3',   'text_batched'),
    ]

    rows = []
    for fname, default_hw, default_mod in file_configs:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        for rec in data:
            energy = rec.get('total_energy_j')
            gt = rec.get('generation_time_sec') or rec.get('generation_time_s')
            if energy is None or gt is None or energy <= 0 or gt <= 0:
                continue

            hw = rec.get('hardware', default_hw)
            if hw not in HARDWARE_SPECS:
                continue
            mod = rec.get('modality', default_mod)
            mn  = rec.get('model', '')
            if mn in MODEL_INFO:
                arch = MODEL_INFO[mn]['arch']
                pB   = MODEL_INFO[mn]['params_B']
            else:
                pB = rec.get('params_B', 0)
                if pB == 0:
                    pB = rec.get('params_M', 0) / 1000.0
                arch = 'AR' if mod in ('text','text_batched','music') else 'Diff'

            sp = HARDWARE_SPECS[hw]
            steps = rec.get('steps', 1)
            if mod in ('text','text_batched','music'):
                steps = 1
            bs = rec.get('batch_size', 1)
            oc = compute_output_complexity(rec, mod)

            rows.append({
                'model': mn, 'modality': mod, 'hardware': hw,
                'arch_type': 1 if arch == 'Diff' else 0,
                'params_B': pB,
                'hw_tdp':  sp['tdp'],
                'compute_units': sp['compute_units'],
                'memory_bw': sp['memory_bw'],
                'output_complexity': oc,
                'num_steps': steps,
                'batch_size': bs,
                'energy_j': energy, 'time_s': gt,
                'avg_power_w': rec.get('avg_power_w', energy / gt),
            })

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} individual records across {df['hardware'].nunique()} HW, "
          f"{df['model'].nunique()} models, {df['modality'].nunique()} modalities")
    print(df.groupby('hardware').size().to_string())
    return df


def aggregate_to_config_means(df):
    """Aggregate individual repetitions to config-level means.
    
    Config is defined by (model, hardware, modality, arch_type, params_B,
    hw_tdp, compute_units, memory_bw, output_complexity, num_steps, batch_size).
    
    This eliminates data leakage from repeated measurements of the same
    configuration appearing in both train and test folds.
    """
    config_cols = ['model', 'hardware', 'modality', 'arch_type', 'params_B',
                   'hw_tdp', 'compute_units', 'memory_bw',
                   'output_complexity', 'num_steps', 'batch_size']
    target_cols = ['energy_j', 'time_s', 'avg_power_w']
    
    agg = df.groupby(config_cols, as_index=False).agg(
        energy_j=('energy_j', 'mean'),
        time_s=('time_s', 'mean'),
        avg_power_w=('avg_power_w', 'mean'),
        energy_std=('energy_j', 'std'),
        time_std=('time_s', 'std'),
        n_reps=('energy_j', 'count'),
    )
    agg['energy_std'] = agg['energy_std'].fillna(0)
    agg['time_std'] = agg['time_std'].fillna(0)
    
    print(f"\nAggregated {len(df)} individual records → {len(agg)} unique configurations")
    print(f"  Mean repetitions per config: {agg['n_reps'].mean():.1f}")
    print(f"  Median repetitions per config: {agg['n_reps'].median():.0f}")
    print(f"  Energy range: {agg['energy_j'].min():.1f} – {agg['energy_j'].max():.1f} J")
    print(f"  Time range: {agg['time_s'].min():.2f} – {agg['time_s'].max():.2f} s")
    return agg


def compute_coverage_matrix(df):
    """Compute model×hardware coverage matrix."""
    coverage = {}
    for model in sorted(df['model'].unique()):
        coverage[model] = {}
        for hw in sorted(df['hardware'].unique(), key=lambda x: HARDWARE_SPECS[x]['tdp']):
            n = len(df[(df['model'] == model) & (df['hardware'] == hw)])
            coverage[model][HW_SHORT[hw]] = n
    return coverage


# ============================================================
# FEATURES
# ============================================================
RAW_FEATURES = ['params_B','arch_type','hw_tdp','compute_units',
                'memory_bw','output_complexity','num_steps','batch_size']

LOG_FEATURES = ['log_params','log_tdp','log_compute','log_membw',
                'log_complexity','log_steps','log_batch','arch_type']

def add_log_features(df):
    df = df.copy()
    df['log_params']     = np.log1p(df['params_B'])
    df['log_tdp']        = np.log1p(df['hw_tdp'])
    df['log_compute']    = np.log1p(df['compute_units'])
    df['log_membw']      = np.log1p(df['memory_bw'])
    df['log_complexity'] = np.log1p(df['output_complexity'])
    df['log_steps']      = np.log1p(df['num_steps'])
    df['log_batch']      = np.log1p(df['batch_size'])
    df['log_energy']     = np.log1p(df['energy_j'])
    df['log_time']       = np.log1p(df['time_s'])
    return df


def metrics(yt, yp):
    r2 = r2_score(yt, yp)
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mask = yt > 0
    mape = float(np.mean(np.abs((yt[mask]-yp[mask])/yt[mask]))*100)
    return {'R2': round(r2,4), 'RMSE': round(rmse,4), 'MAPE': round(mape,2)}


def get_models():
    return {
        'Linear': LinearRegression(),
        'Ridge':  Ridge(alpha=1.0),
        'RF':     RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.1,
                                      random_state=42, verbosity=0),
        'SVR':    SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1),
    }


# ============================================================
# CV ROUTINES
# ============================================================
def kfold_cv(df, feature_cols, target_col, k=5):
    """5-fold CV on config-level means (no data leakage)."""
    X = df[feature_cols].values.astype(float)
    y = df[target_col].values
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    res = {}
    fold_res = {}
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    for name, mdl in get_models().items():
        Xi = Xs if name == 'SVR' else X
        # Compute per-fold metrics for std
        fold_r2s = []
        fold_mapes = []
        yp_all = np.zeros_like(y)
        for train_idx, test_idx in kf.split(Xi):
            Xtr, Xte = Xi[train_idx], Xi[test_idx]
            ytr, yte = y[train_idx], y[test_idx]
            from sklearn.base import clone
            m = clone(mdl)
            m.fit(Xtr, ytr)
            yp = m.predict(Xte)
            yp_all[test_idx] = yp
            fm = metrics(yte, yp)
            fold_r2s.append(fm['R2'])
            fold_mapes.append(fm['MAPE'])
        
        m_overall = metrics(y, yp_all)
        res[name] = m_overall
        fold_res[name] = {
            'R2_std': round(float(np.std(fold_r2s)), 4),
            'MAPE_std': round(float(np.std(fold_mapes)), 2),
            'R2_per_fold': [round(float(x), 4) for x in fold_r2s],
            'MAPE_per_fold': [round(float(x), 2) for x in fold_mapes],
        }
        print(f"  {name:10s}: R²={m_overall['R2']:.4f}±{np.std(fold_r2s):.4f}  "
              f"RMSE={m_overall['RMSE']:.2f}  MAPE={m_overall['MAPE']:.1f}±{np.std(fold_mapes):.1f}%")
    return res, fold_res


def kfold_cv_logspace(df, raw_target_col, k=5):
    """5-fold CV: train in log-space, evaluate in raw space.
    
    This gives the most honest within-platform accuracy metric
    because log-space training handles the multi-scale energy range
    properly, and evaluation in raw space gives directly interpretable
    MAPE and RMSE numbers.
    """
    df2 = add_log_features(df)
    feat = LOG_FEATURES
    log_target = 'log_' + raw_target_col.replace('_j', '').replace('_s', '')
    if raw_target_col == 'energy_j':
        log_target = 'log_energy'
    elif raw_target_col == 'time_s':
        log_target = 'log_time'
    
    X = df2[feat].values.astype(float)
    y_log = df2[log_target].values
    y_raw = df[raw_target_col].values
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    res = {}
    fold_res = {}
    
    model_factories = {
        'Linear': lambda: LinearRegression(),
        'Ridge':  lambda: Ridge(alpha=1.0),
        'RF':     lambda: RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        'XGBoost': lambda: xgb.XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.1,
                                              random_state=42, verbosity=0),
        'SVR':    lambda: SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1),
    }
    
    for name, factory in model_factories.items():
        fold_r2s = []
        fold_mapes = []
        yp_raw_all = np.zeros_like(y_raw)
        
        for train_idx, test_idx in kf.split(X):
            Xtr, Xte = X[train_idx], X[test_idx]
            ytr_log, yte_log = y_log[train_idx], y_log[test_idx]
            yte_raw = y_raw[test_idx]
            
            scaler = StandardScaler()
            Xtrs = scaler.fit_transform(Xtr)
            Xtes = scaler.transform(Xte)
            
            mdl = factory()
            if name == 'SVR':
                mdl.fit(Xtrs, ytr_log)
                yp_log = mdl.predict(Xtes)
            else:
                mdl.fit(Xtr, ytr_log)
                yp_log = mdl.predict(Xte)
            
            yp_raw = np.expm1(yp_log)
            yp_raw = np.maximum(yp_raw, 0.01)
            yp_raw_all[test_idx] = yp_raw
            
            fm = metrics(yte_raw, yp_raw)
            fold_r2s.append(fm['R2'])
            fold_mapes.append(fm['MAPE'])
        
        m_overall = metrics(y_raw, yp_raw_all)
        res[name] = m_overall
        fold_res[name] = {
            'R2_std': round(float(np.std(fold_r2s)), 4),
            'MAPE_std': round(float(np.std(fold_mapes)), 2),
            'R2_per_fold': [round(float(x), 4) for x in fold_r2s],
            'MAPE_per_fold': [round(float(x), 2) for x in fold_mapes],
        }
        print(f"  {name:10s}: R²={m_overall['R2']:.4f}±{np.std(fold_r2s):.4f}  "
              f"RMSE={m_overall['RMSE']:.2f}  MAPE={m_overall['MAPE']:.1f}±{np.std(fold_mapes):.1f}%")
    return res, fold_res


def loho_cv(df, feature_cols, target_col):
    """Leave-One-Hardware-Out CV. The KEY experiment."""
    X = df[feature_cols].values.astype(float)
    y = df[target_col].values
    hw_arr = df['hardware'].values
    hw_list = df['hardware'].unique()

    res_overall = {}
    res_per_hw = {}

    model_factories = {
        'Linear':  lambda: LinearRegression(),
        'Ridge':   lambda: Ridge(alpha=1.0),
        'RF':      lambda: RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        'XGBoost': lambda: xgb.XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.1,
                                              random_state=42, verbosity=0),
        'SVR':     lambda: SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1),
    }

    for name, factory in model_factories.items():
        alls_t, alls_p = [], []
        hw_m = {}
        for test_hw in hw_list:
            tr = hw_arr != test_hw
            te = hw_arr == test_hw
            Xtr, Xte = X[tr], X[te]
            ytr, yte = y[tr], y[te]
            scaler = StandardScaler()
            Xtrs = scaler.fit_transform(Xtr)
            Xtes = scaler.transform(Xte)
            mdl = factory()
            if name == 'SVR':
                mdl.fit(Xtrs, ytr); yp = mdl.predict(Xtes)
            else:
                mdl.fit(Xtr, ytr); yp = mdl.predict(Xte)
            alls_t.extend(yte); alls_p.extend(yp)
            hw_m[test_hw] = metrics(yte, yp)

        overall = metrics(np.array(alls_t), np.array(alls_p))
        res_overall[name] = overall
        res_per_hw[name] = hw_m
        print(f"  {name:10s}: R²={overall['R2']:.4f}  MAPE={overall['MAPE']:.1f}%")
        for h, m in hw_m.items():
            print(f"    → {HW_SHORT[h]:8s}: R²={m['R2']:.4f}  MAPE={m['MAPE']:.1f}%")

    return res_overall, res_per_hw


def lomo_cv(df, feature_cols, target_col):
    """Leave-One-Model-Out CV."""
    X = df[feature_cols].values.astype(float)
    y = df[target_col].values
    mdl_arr = df['model'].values
    mdl_list = df['model'].unique()

    model_factories = {
        'Linear':  lambda: LinearRegression(),
        'Ridge':   lambda: Ridge(alpha=1.0),
        'RF':      lambda: RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        'XGBoost': lambda: xgb.XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.1,
                                              random_state=42, verbosity=0),
        'SVR':     lambda: SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1),
    }

    res = {}; res_per = {}
    for name, factory in model_factories.items():
        alls_t, alls_p = [], []
        per = {}
        for test_m in mdl_list:
            tr = mdl_arr != test_m; te = mdl_arr == test_m
            Xtr, Xte = X[tr], X[te]; ytr, yte = y[tr], y[te]
            sc = StandardScaler(); Xtrs = sc.fit_transform(Xtr); Xtes = sc.transform(Xte)
            mdl = factory()
            if name == 'SVR':
                mdl.fit(Xtrs, ytr); yp = mdl.predict(Xtes)
            else:
                mdl.fit(Xtr, ytr); yp = mdl.predict(Xte)
            alls_t.extend(yte); alls_p.extend(yp)
            per[test_m] = metrics(yte, yp)
        overall = metrics(np.array(alls_t), np.array(alls_p))
        res[name] = overall; res_per[name] = per
        print(f"  {name:10s}: R²={overall['R2']:.4f}  MAPE={overall['MAPE']:.1f}%")
        for mm, m in per.items():
            print(f"    → {mm:25s}: R²={m['R2']:.4f}  MAPE={m['MAPE']:.1f}%")
    return res, res_per


# ============================================================
# LOG-LINEAR APPROACH (best for LOHO)
# ============================================================
def loho_loglinear(df):
    """Log-linear model with LOHO-CV — expected to generalize better."""
    df2 = add_log_features(df)
    feat = LOG_FEATURES
    X = df2[feat].values.astype(float)
    y_e = df2['log_energy'].values
    y_t = df2['log_time'].values
    y_raw_e = df2['energy_j'].values
    y_raw_t = df2['time_s'].values
    hw_arr = df2['hardware'].values
    hw_list = df2['hardware'].unique()

    hw_me = {}; hw_mt = {}
    all_te, all_pe, all_tt, all_pt = [], [], [], []

    for test_hw in hw_list:
        tr = hw_arr != test_hw; te = hw_arr == test_hw
        lr_e = LinearRegression().fit(X[tr], y_e[tr])
        lr_t = LinearRegression().fit(X[tr], y_t[tr])
        pe = np.expm1(lr_e.predict(X[te]))
        pt = np.expm1(lr_t.predict(X[te]))
        pe = np.maximum(pe, 0.01)
        pt = np.maximum(pt, 0.01)
        all_te.extend(y_raw_e[te]); all_pe.extend(pe)
        all_tt.extend(y_raw_t[te]); all_pt.extend(pt)
        hw_me[test_hw] = metrics(y_raw_e[te], pe)
        hw_mt[test_hw] = metrics(y_raw_t[te], pt)
        print(f"    → {HW_SHORT[test_hw]:8s}: Energy MAPE={hw_me[test_hw]['MAPE']:.1f}%  Time MAPE={hw_mt[test_hw]['MAPE']:.1f}%")

    oe = metrics(np.array(all_te), np.array(all_pe))
    ot = metrics(np.array(all_tt), np.array(all_pt))
    print(f"    Overall: Energy R²={oe['R2']:.4f} MAPE={oe['MAPE']:.1f}%, Time R²={ot['R2']:.4f} MAPE={ot['MAPE']:.1f}%")

    # Also fit on full data to get the formula coefficients
    lr_full_e = LinearRegression().fit(X, y_e)
    lr_full_t = LinearRegression().fit(X, y_t)

    # Compute in-sample R² on raw energy
    pe_full = np.expm1(lr_full_e.predict(X))
    pe_full = np.maximum(pe_full, 0.01)
    insample_r2 = r2_score(y_raw_e, pe_full)
    insample_metrics = metrics(y_raw_e, pe_full)
    print(f"\n  Full-data in-sample fit: R²={insample_r2:.4f}, MAPE={insample_metrics['MAPE']:.1f}%")

    print("\n  Full-data Energy formula: log1p(E) = {:.3f}".format(lr_full_e.intercept_))
    for fn, c in zip(feat, lr_full_e.coef_):
        print(f"    + {c:+.4f} * {fn}")

    print("\n  Full-data Time formula: log1p(T) = {:.3f}".format(lr_full_t.intercept_))
    for fn, c in zip(feat, lr_full_t.coef_):
        print(f"    + {c:+.4f} * {fn}")

    return {
        'loho_energy_overall': oe,
        'loho_time_overall': ot,
        'loho_energy_per_hw': hw_me,
        'loho_time_per_hw': hw_mt,
        'insample_energy_r2': round(float(insample_r2), 4),
        'insample_energy_metrics': insample_metrics,
        'energy_coefs': {fn: round(float(c),4) for fn,c in zip(feat, lr_full_e.coef_)},
        'energy_intercept': round(float(lr_full_e.intercept_),4),
        'time_coefs': {fn: round(float(c),4) for fn,c in zip(feat, lr_full_t.coef_)},
        'time_intercept': round(float(lr_full_t.intercept_),4),
    }


# ============================================================
# LOG-SPACE LOHO for ML models
# ============================================================
def loho_logspace(df):
    """Train ML models in log-space for LOHO — should generalize better."""
    df2 = add_log_features(df)
    feat = LOG_FEATURES
    X = df2[feat].values.astype(float)
    y_log = df2['log_energy'].values
    y_raw = df2['energy_j'].values
    hw_arr = df2['hardware'].values
    hw_list = df2['hardware'].unique()

    model_factories = {
        'Ridge(log)': lambda: Ridge(alpha=1.0),
        'RF(log)':    lambda: RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        'XGB(log)':   lambda: xgb.XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.1,
                                                  random_state=42, verbosity=0),
    }

    results = {}
    per_hw = {}
    for name, factory in model_factories.items():
        alls_t, alls_p = [], []
        hm = {}
        for test_hw in hw_list:
            tr = hw_arr != test_hw; te = hw_arr == test_hw
            mdl = factory()
            mdl.fit(X[tr], y_log[tr])
            yp_log = mdl.predict(X[te])
            yp = np.expm1(yp_log)
            yp = np.maximum(yp, 0.01)
            alls_t.extend(y_raw[te]); alls_p.extend(yp)
            hm[test_hw] = metrics(y_raw[te], yp)
        overall = metrics(np.array(alls_t), np.array(alls_p))
        results[name] = overall
        per_hw[name] = hm
        print(f"  {name:12s}: R²={overall['R2']:.4f}  MAPE={overall['MAPE']:.1f}%")
        for h, m in hm.items():
            print(f"    → {HW_SHORT[h]:8s}: R²={m['R2']:.4f}  MAPE={m['MAPE']:.1f}%")

    return results, per_hw


# ============================================================
# SHAP ANALYSIS
# ============================================================
def run_shap(df):
    df2 = add_log_features(df)
    feat = LOG_FEATURES
    nice = {
        'log_params': 'Model Size',
        'log_tdp': 'HW TDP',
        'log_compute': 'Compute Units',
        'log_membw': 'Mem. Bandwidth',
        'log_complexity': 'Output Complexity',
        'log_steps': 'Diff. Steps',
        'log_batch': 'Batch Size',
        'arch_type': 'Arch. Type (Diff)'
    }

    X = df2[feat].values.astype(float)
    y = df2['log_energy'].values

    mdl = xgb.XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.1,
                             random_state=42, verbosity=0)
    mdl.fit(X, y)
    expl = shap.TreeExplainer(mdl)
    sv = expl.shap_values(X)
    mean_abs = np.abs(sv).mean(axis=0)

    imp = {feat[i]: float(mean_abs[i]) for i in range(len(feat))}
    print("\n  Overall SHAP (log-energy):")
    for fn, v in sorted(imp.items(), key=lambda x: -x[1]):
        print(f"    {nice.get(fn,fn):22s}: {v:.3f}")

    # Per-architecture
    ar_mask = df2['arch_type'] == 0
    diff_mask = df2['arch_type'] == 1
    arch_imps = {}
    for label, mask in [('AR', ar_mask), ('Diffusion', diff_mask)]:
        Xs = X[mask]; ys = y[mask]
        if len(ys) < 10: continue
        m2 = xgb.XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.1,
                                random_state=42, verbosity=0)
        m2.fit(Xs, ys)
        e2 = shap.TreeExplainer(m2)
        s2 = e2.shap_values(Xs)
        ma2 = np.abs(s2).mean(axis=0)
        ai = {feat[i]: float(ma2[i]) for i in range(len(feat))}
        arch_imps[label] = ai
        print(f"\n  SHAP for {label}:")
        for fn, v in sorted(ai.items(), key=lambda x: -x[1]):
            print(f"    {nice.get(fn,fn):22s}: {v:.3f}")

    # === FIGURE (improved readability) ===
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Determine global max for consistent x-axis
    all_vals = list(imp.values())
    for ai in arch_imps.values():
        all_vals.extend(ai.values())
    xmax = max(all_vals) * 1.15

    for idx, (title, imp_dict) in enumerate([
        ('(a) All Models', imp),
        ('(b) Autoregressive', arch_imps.get('AR', {})),
        ('(c) Diffusion', arch_imps.get('Diffusion', {})),
    ]):
        ax = axes[idx]
        if not imp_dict:
            continue
        names = [nice.get(k,k) for k in feat]
        vals  = [imp_dict.get(k,0) for k in feat]
        si = np.argsort(vals)
        colors = ['#2196F3','#FF9800','#4CAF50'][idx]
        ax.barh(range(len(feat)), [vals[i] for i in si], color=colors)
        ax.set_yticks(range(len(feat)))
        ax.set_yticklabels([names[i] for i in si], fontsize=9)
        ax.set_xlabel('Mean |SHAP|', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlim(0, xmax)
        ax.tick_params(axis='x', labelsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'shap_importance.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(FIG_DIR, 'shap_importance.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print("\n  Saved: shap_importance.pdf")

    return imp, arch_imps


# ============================================================
# FIGURES
# ============================================================
def plot_loho_scatter(df):
    """Predicted vs actual scatter for LOHO-CV, best model (XGBoost in log-space).
    Single-panel (energy only) for column-width figure readability."""
    df2 = add_log_features(df)
    feat = LOG_FEATURES
    X = df2[feat].values.astype(float)
    y_log_e = df2['log_energy'].values
    y_raw_e = df2['energy_j'].values
    hw_arr = df2['hardware'].values
    hw_list = df2['hardware'].unique()
    colors = {'Apple M4 Pro': '#FF6B6B', 'NVIDIA A100-SXM4-40GB': '#4ECDC4', 'NVIDIA H100 80GB HBM3': '#1A535C'}
    markers = {'Apple M4 Pro': 'o', 'NVIDIA A100-SXM4-40GB': 's', 'NVIDIA H100 80GB HBM3': '^'}

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4))

    for test_hw in hw_list:
        tr = hw_arr != test_hw; te = hw_arr == test_hw
        mdl = xgb.XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.1,
                                random_state=42, verbosity=0)
        mdl.fit(X[tr], y_log_e[tr])
        yp = np.expm1(mdl.predict(X[te]))
        yp = np.maximum(yp, 0.01)
        ax.scatter(y_raw_e[te], yp, alpha=0.5, s=18,
                  color=colors[test_hw], marker=markers[test_hw],
                  label=f'{HW_SHORT[test_hw]} (held out)')

    lims = [0.5, max(y_raw_e)*2]
    ax.plot(lims, lims, 'k--', alpha=0.5, lw=1)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Measured Energy (J)', fontsize=10)
    ax.set_ylabel('Predicted Energy (J)', fontsize=10)
    ax.set_title('LOHO-CV: Energy Prediction', fontsize=11)
    ax.legend(fontsize=8, loc='upper left')
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'loho_scatter.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(FIG_DIR, 'loho_scatter.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print("  Saved: loho_scatter.pdf")


def plot_comparison_bar(kf, loho, lomo, loho_log, analytical_loho):
    """Model comparison across CV strategies."""
    model_names = ['Ridge', 'RF', 'XGBoost', 'SVR']

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(model_names))
    w = 0.2

    # R² panel
    ax = axes[0]
    r2_kf   = [kf[m]['R2'] for m in model_names]
    r2_loho = [loho[m]['R2'] for m in model_names]
    r2_lomo = [lomo[m]['R2'] for m in model_names]
    r2_ll = []
    for m in model_names:
        key = m + '(log)'
        if key in loho_log:
            r2_ll.append(loho_log[key]['R2'])
        else:
            r2_ll.append(None)

    bars1 = ax.bar(x - 1.5*w, r2_kf,   w, label='5-Fold CV', color='#2196F3')
    bars2 = ax.bar(x - 0.5*w, r2_loho,  w, label='LOHO (raw)', color='#FF9800')
    r2_ll_plot = [v if v is not None else 0 for v in r2_ll]
    bars3 = ax.bar(x + 0.5*w, r2_ll_plot, w, label='LOHO (log)', color='#9C27B0')
    bars4 = ax.bar(x + 1.5*w, r2_lomo,  w, label='LOMO', color='#4CAF50')
    ax.set_ylabel('R²'); ax.set_title('(a) R² Score')
    ax.set_xticks(x); ax.set_xticklabels(model_names)
    ax.legend(fontsize=7)
    ax.axhline(y=0, color='gray', ls=':', lw=0.5)

    # MAPE panel
    ax = axes[1]
    mape_kf   = [kf[m]['MAPE'] for m in model_names]
    mape_loho = [loho[m]['MAPE'] for m in model_names]
    mape_lomo = [lomo[m]['MAPE'] for m in model_names]
    mape_ll = []
    for m in model_names:
        key = m + '(log)'
        if key in loho_log:
            mape_ll.append(loho_log[key]['MAPE'])
        else:
            mape_ll.append(0)
    ax.bar(x - 1.5*w, mape_kf,   w, label='5-Fold CV', color='#2196F3')
    ax.bar(x - 0.5*w, mape_loho, w, label='LOHO (raw)', color='#FF9800')
    ax.bar(x + 0.5*w, mape_ll,   w, label='LOHO (log)', color='#9C27B0')
    ax.bar(x + 1.5*w, mape_lomo, w, label='LOMO', color='#4CAF50')
    ax.set_ylabel('MAPE (%)'); ax.set_title('(b) Mean Absolute Percentage Error')
    ax.set_xticks(x); ax.set_xticklabels(model_names)
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'model_comparison.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(FIG_DIR, 'model_comparison.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print("  Saved: model_comparison.pdf")


def plot_loho_per_hw(per_hw_raw, per_hw_log, analytical_per_hw):
    """Per-hardware MAPE bar chart."""
    hw_list = ['Apple M4 Pro', 'NVIDIA A100-SXM4-40GB', 'NVIDIA H100 80GB HBM3']
    hw_labels = [HW_SHORT[h] for h in hw_list]

    methods = []
    for m in ['Ridge', 'XGBoost']:
        mapes = [per_hw_raw[m][h]['MAPE'] for h in hw_list]
        methods.append((m + ' (raw)', mapes))
    for m in ['Ridge(log)', 'XGB(log)']:
        mapes = [per_hw_log[m][h]['MAPE'] for h in hw_list]
        methods.append((m, mapes))
    mapes = [analytical_per_hw[h]['MAPE'] for h in hw_list]
    methods.append(('Log-Linear', mapes))

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(hw_labels))
    w = 0.15
    colors_list = ['#FF9800', '#E91E63', '#9C27B0', '#3F51B5', '#009688']
    for i, (label, mapes) in enumerate(methods):
        offset = (i - len(methods)/2 + 0.5) * w
        ax.bar(x + offset, mapes, w, label=label, color=colors_list[i % len(colors_list)])

    ax.set_xlabel('Test Hardware (Held Out)')
    ax.set_ylabel('MAPE (%)')
    ax.set_title('Leave-One-Hardware-Out: Per-Platform Energy Prediction Error')
    ax.set_xticks(x); ax.set_xticklabels(hw_labels)
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'loho_per_hardware.pdf'), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(FIG_DIR, 'loho_per_hardware.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print("  Saved: loho_per_hardware.pdf")


# ============================================================
# MAIN
# ============================================================
def main():
    print("="*70)
    print("PREDICTIVE COST MODEL — TCAS-II EXPRESS BRIEFS (v2: config means)")
    print("="*70)

    # Load individual records
    df_raw = load_all_data()
    df_raw.to_csv(os.path.join(OUT_DIR, 'unified_dataset_raw.csv'), index=False)
    n_raw = len(df_raw)
    
    # Coverage matrix (from raw data)
    coverage = compute_coverage_matrix(df_raw)
    print("\nCoverage matrix (individual records):")
    for model, hw_counts in coverage.items():
        print(f"  {model:25s}: {hw_counts}")

    # Aggregate to config-level means (fixes data leakage)
    df = aggregate_to_config_means(df_raw)
    df.to_csv(os.path.join(OUT_DIR, 'unified_dataset.csv'), index=False)
    
    n_configs = len(df)
    n_hw = int(df['hardware'].nunique())
    n_models = int(df['model'].nunique())
    n_modalities = int(df['modality'].nunique())
    
    print(f"\nEnergy range: {df['energy_j'].min():.1f} – {df['energy_j'].max():.1f} J")
    print(f"Time range: {df['time_s'].min():.2f} – {df['time_s'].max():.2f} s")
    print(f"Configs per hardware:")
    print(df.groupby('hardware').size().to_string())

    # ---- 5-Fold CV (raw features, config means) ----
    print("\n" + "="*50)
    print("[1] 5-Fold CV — Energy (raw features, config means)")
    print("="*50)
    kf_e, kf_e_folds = kfold_cv(df, RAW_FEATURES, 'energy_j')

    print("\n[1b] 5-Fold CV — Time")
    kf_t, kf_t_folds = kfold_cv(df, RAW_FEATURES, 'time_s')

    # ---- 5-Fold CV (log features, evaluated in raw energy space) ----
    print("\n[1c] 5-Fold CV — Log-space training, raw-space evaluation (Energy)")
    kf_log_raw, kf_log_raw_folds = kfold_cv_logspace(df, 'energy_j')

    # ---- LOHO CV (raw) ----
    print("\n" + "="*50)
    print("[2] LOHO-CV — Energy (raw features) ★ KEY EXPERIMENT")
    print("="*50)
    loho_e, loho_e_hw = loho_cv(df, RAW_FEATURES, 'energy_j')

    # ---- LOHO CV (log-space ML) ----
    print("\n" + "="*50)
    print("[3] LOHO-CV — Energy (log-space ML models)")
    print("="*50)
    loho_log, loho_log_hw = loho_logspace(df)

    # ---- LOHO Log-Linear Analytical ----
    print("\n" + "="*50)
    print("[4] LOHO-CV — Log-Linear Analytical Formula")
    print("="*50)
    analytical = loho_loglinear(df)

    # ---- LOMO CV ----
    print("\n" + "="*50)
    print("[5] LOMO-CV — Energy (raw features)")
    print("="*50)
    lomo_e, lomo_e_per = lomo_cv(df, RAW_FEATURES, 'energy_j')

    # ---- SHAP ----
    print("\n" + "="*50)
    print("[6] SHAP Feature Importance")
    print("="*50)
    shap_imp, shap_arch = run_shap(df)

    # ---- Figures ----
    print("\n" + "="*50)
    print("[7] Generating Figures")
    print("="*50)
    plot_loho_scatter(df)
    plot_comparison_bar(kf_log_raw, loho_e, lomo_e, loho_log, analytical['loho_energy_overall'])
    plot_loho_per_hw(loho_e_hw, loho_log_hw, analytical['loho_energy_per_hw'])

    # ---- Save ----
    # Coverage matrix with short names
    coverage_short = {}
    for model, hw_counts in coverage.items():
        coverage_short[model] = hw_counts

    all_results = {
        'dataset': {
            'n_raw_records': n_raw,
            'n_configs': n_configs,
            'n_hw': n_hw,
            'n_models': n_models,
            'n_modalities': n_modalities,
            'mean_reps_per_config': round(float(df_raw.groupby(
                ['model','hardware','modality','output_complexity','num_steps','batch_size']
            ).size().mean()), 1),
        },
        'coverage_matrix': coverage_short,
        'kfold_energy_raw': kf_e,
        'kfold_energy_raw_folds': kf_e_folds,
        'kfold_energy_logspace': kf_log_raw,
        'kfold_energy_logspace_folds': kf_log_raw_folds,
        'kfold_time': kf_t,
        'loho_energy_raw': loho_e,
        'loho_energy_raw_per_hw': {m: {HW_SHORT[h]: v for h,v in d.items()} for m,d in loho_e_hw.items()},
        'loho_energy_logspace': loho_log,
        'loho_energy_logspace_per_hw': {m: {HW_SHORT[h]: v for h,v in d.items()} for m,d in loho_log_hw.items()},
        'analytical': analytical,
        'lomo_energy': lomo_e,
        'lomo_energy_per_model': {
            name: {model: vals for model, vals in per.items()}
            for name, per in lomo_e_per.items()
        },
        'shap_overall': {k: round(float(v),4) for k,v in shap_imp.items()},
        'shap_per_arch': {a: {k: round(float(v),4) for k,v in d.items()} for a,d in shap_arch.items()},
    }

    with open(os.path.join(OUT_DIR, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda o: round(float(o),4) if hasattr(o,'item') else str(o))

    print("\n" + "="*70)
    print("DONE. Results:", OUT_DIR)
    print("Figures:", FIG_DIR)
    print("="*70)


if __name__ == '__main__':
    main()
