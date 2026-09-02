from __future__ import annotations
import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from PIL import Image, ImageDraw
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.nonparametric.smoothers_lowess import lowess
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
RANDOM_STATE = 42
CORE = ['age', 'history_hf', 'history_af', 'ph', 'urea', 'lactate']
MISS = ['ph_missing', 'urea_missing', 'lactate_missing']
RESP = ['resp_support']
EMBEDDED_MEDIANS = {'age': 72.0, 'history_hf': 0.0, 'history_af': 0.0, 'ph': 7.39, 'urea': 34.05, 'lactate': 1.2}

def set_plot_style() -> None:
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 15, 'axes.labelsize': 12, 'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5, 'legend.fontsize': 9.5, 'axes.linewidth': 0.9, 'figure.dpi': 160, 'savefig.dpi': 300, 'legend.frameon': False})

def find_file(folder: Path, base: str, required: bool=True) -> Path | None:
    candidates = []
    for suffix in ['.csv.gz', '.csv', '.CSV.GZ', '.CSV']:
        candidates.append(folder / f'{base}{suffix}')
    for suffix in ['.csv.gz', '.csv', '.CSV.GZ', '.CSV']:
        candidates.extend(folder.rglob(f'{base}{suffix}'))
    seen = set()
    unique = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    for p in unique:
        if p.exists():
            return p
    if required:
        nearby = '\n'.join((str(p) for p in folder.rglob('*.csv*') if p.is_file()))
        raise FileNotFoundError(f'Could not find {base}.csv.gz or {base}.csv under {folder}. Found:\n{nearby[:4000]}')
    return None

def read_csv(path: Path) -> pd.DataFrame:
    print(f'Reading: {path}')
    return pd.read_csv(path, low_memory=False)

def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out

def clean_text(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.strip()

def first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    m = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in m:
            return m[n.lower()]
    return None

def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, n_boot: int=2000, seed: int=42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.unique(y[idx]).size < 2:
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

def bootstrap_metric_ci(y: np.ndarray, p: np.ndarray, metric, n_boot: int=1000, seed: int=42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.unique(y[idx]).size < 2:
            continue
        vals.append(metric(y[idx], p[idx]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

def prepare_age(patient: pd.DataFrame) -> pd.Series:
    age = patient['age'].astype(str).str.strip().replace({'> 89': '90', '>89': '90', '90+': '90'})
    return pd.to_numeric(age, errors='coerce')

def load_medians(path_or_dir: Path | None) -> dict[str, float]:
    if path_or_dir is None:
        print('No medians path provided; using embedded frozen internal medians: age=72.0, history_hf=0.0, history_af=0.0, ph=7.39, urea=34.05, lactate=1.2.')
        return EMBEDDED_MEDIANS.copy()
    p = Path(path_or_dir)
    candidates = []
    if p.is_file():
        candidates = [p]
    elif p.is_dir():
        candidates = [p / 'original_model_medians.csv', p / 'internal_medians.csv', p / 'development_medians.csv', p / 'model_medians.csv'] + list(p.rglob('*median*.csv'))
    for c in candidates:
        if c.exists():
            try:
                df = norm_cols(pd.read_csv(c))
                if {'variable', 'median'}.issubset(df.columns):
                    df['variable'] = df['variable'].astype(str).str.strip()
                    df['median'] = pd.to_numeric(df['median'], errors='coerce')
                    d = dict(zip(df['variable'], df['median']))
                    if all((k in d and pd.notna(d[k]) for k in CORE)):
                        print(f'Using medians file: {c}')
                        return {k: float(d[k]) for k in CORE}
            except Exception:
                pass
    print('No usable medians file found. Using embedded frozen internal medians: age=72.0, history_hf=0.0, history_af=0.0, ph=7.39, urea=34.05, lactate=1.2.')
    return EMBEDDED_MEDIANS.copy()

def extract_aecopd_ids(diagnosis: pd.DataFrame, exclude_competing: bool=False) -> set[int]:
    text = clean_text(diagnosis['diagnosisstring'])
    copd = text.str.contains('\\bcopd\\b|chronic obstructive pulmonary', regex=True, na=False)
    acute = text.str.contains('exacerb|acute|respiratory failure|bronchitis|bronchospasm', regex=True, na=False)
    ids = set(diagnosis.loc[copd & acute, 'patientunitstayid'].dropna().astype(int))
    if not ids:
        ids = set(diagnosis.loc[copd, 'patientunitstayid'].dropna().astype(int))
    if exclude_competing:
        competing = text.str.contains('pneumonia|sepsis|aspiration', regex=True, na=False)
        bad = set(diagnosis.loc[competing, 'patientunitstayid'].dropna().astype(int))
        ids = ids - bad
    print(f'AECOPD candidate stays: {len(ids)}')
    return ids

def extract_events(diagnosis: pd.DataFrame, ids: set[int]) -> pd.DataFrame:
    dx = diagnosis[diagnosis['patientunitstayid'].isin(ids)].copy()
    text = clean_text(dx['diagnosisstring'])
    chronic = text.str.contains('history of|past history|hx of|chronic|known|baseline|prior|previous|old ', regex=True, na=False)
    mi = text.str.contains('myocardial infarction|nstemi|stemi|acute mi|non.?st elevation|st elevation', regex=True, na=False) & ~text.str.contains('old myocardial infarction|old mi|history of|hx of|prior|previous', regex=True, na=False)
    pe = text.str.contains('pulmonary embolism|\\bpe\\b', regex=True, na=False) & ~chronic
    pulm_edema = text.str.contains('pulmonary edema|pulmonary oedema|acute pulmonary edema|acute pulmonary oedema', regex=True, na=False) & ~chronic
    arr = text.str.contains('atrial fibrillation|atrial flutter|arrhythmia|tachyarrhythmia|supraventricular tachycardia|\\bsvt\\b|rapid ventricular response|\\brvr\\b', regex=True, na=False) & ~chronic
    hf = text.str.contains('heart failure|cardiac failure|congestive heart failure|\\bchf\\b|decompensated heart failure|fluid overload', regex=True, na=False) & ~chronic
    out = pd.DataFrame({'patientunitstayid': sorted(ids)})
    out['mi_event'] = out['patientunitstayid'].isin(dx.loc[mi, 'patientunitstayid'].dropna().astype(int)).astype(int)
    out['pe_event'] = out['patientunitstayid'].isin(dx.loc[pe, 'patientunitstayid'].dropna().astype(int)).astype(int)
    out['pulmonary_edema_event'] = out['patientunitstayid'].isin(dx.loc[pulm_edema, 'patientunitstayid'].dropna().astype(int)).astype(int)
    out['acute_arrhythmia_event'] = out['patientunitstayid'].isin(dx.loc[arr, 'patientunitstayid'].dropna().astype(int)).astype(int)
    out['hf_decompensation_event'] = out['patientunitstayid'].isin(dx.loc[hf, 'patientunitstayid'].dropna().astype(int)).astype(int)
    cols = ['mi_event', 'pe_event', 'pulmonary_edema_event', 'acute_arrhythmia_event', 'hf_decompensation_event']
    out['cv_event'] = (out[cols].sum(axis=1) > 0).astype(int)
    support = dx.groupby('patientunitstayid')['diagnosisstring'].apply(lambda x: ' | '.join(sorted(set(map(str, x.dropna()))))[:3000]).reset_index(name='diagnosis_strings')
    return out.merge(support, on='patientunitstayid', how='left')

def extract_history(past: pd.DataFrame, ids: set[int]) -> pd.DataFrame:
    hist = pd.DataFrame({'patientunitstayid': sorted(ids)})
    text_cols = [c for c in ['pasthistoryvalue', 'pasthistorypath', 'pasthistorynotetype'] if c in past.columns]
    if text_cols:
        txt = clean_text(past[text_cols[0]])
        for c in text_cols[1:]:
            txt = txt + ' ' + clean_text(past[c])
    else:
        txt = past.astype(str).agg(' '.join, axis=1).str.lower()
    p = past.copy()
    p['txt'] = txt
    p = p[p['patientunitstayid'].isin(ids)].copy()
    hf = p['txt'].str.contains('heart failure|congestive heart failure|cardiac failure|\\bchf\\b', regex=True, na=False)
    af = p['txt'].str.contains('atrial fibrillation|atrial flutter|\\bafib\\b|\\baf\\b', regex=True, na=False)
    hist['history_hf'] = hist['patientunitstayid'].isin(p.loc[hf, 'patientunitstayid'].dropna().astype(int)).astype(int)
    hist['history_af'] = hist['patientunitstayid'].isin(p.loc[af, 'patientunitstayid'].dropna().astype(int)).astype(int)
    return hist

def extract_labs(lab: pd.DataFrame, ids: set[int], start_min: int, end_min: int, strategy: str) -> pd.DataFrame:
    lab = lab[lab['patientunitstayid'].isin(ids)].copy()
    offset_col = first_col(lab, ['labresultoffset', 'labresultrevisedoffset'])
    if offset_col is None:
        raise ValueError('lab file missing labresultoffset/labresultrevisedoffset')
    lab['labname_clean'] = clean_text(lab['labname'])
    lab['lab_value'] = pd.to_numeric(lab['labresult'], errors='coerce')
    lab['lab_offset'] = pd.to_numeric(lab[offset_col], errors='coerce')
    lab = lab.dropna(subset=['lab_value', 'lab_offset'])
    lab = lab[(lab['lab_offset'] >= start_min) & (lab['lab_offset'] <= end_min)].copy()
    ph_mask = lab['labname_clean'].str.fullmatch('ph', na=False) | lab['labname_clean'].str.contains('\\bph\\b', regex=True, na=False)
    lact_mask = lab['labname_clean'].str.contains('lactate|lactic acid', regex=True, na=False)
    bun_mask = lab['labname_clean'].str.contains('\\bbun\\b|blood urea nitrogen', regex=True, na=False)
    urea_direct_mask = lab['labname_clean'].str.contains('\\burea\\b', regex=True, na=False) & ~bun_mask

    def pick(mask: pd.Series, name: str, mode: str) -> pd.DataFrame:
        tmp = lab.loc[mask, ['patientunitstayid', 'lab_offset', 'lab_value', 'labname']].copy()
        if tmp.empty:
            return pd.DataFrame(columns=['patientunitstayid', name, f'{name}_offset_min', f'{name}_labname'])
        if mode == 'first':
            idx = tmp.sort_values(['patientunitstayid', 'lab_offset']).groupby('patientunitstayid').head(1).index
        elif mode == 'worst_min':
            idx = tmp.groupby('patientunitstayid')['lab_value'].idxmin()
        elif mode == 'worst_max':
            idx = tmp.groupby('patientunitstayid')['lab_value'].idxmax()
        else:
            raise ValueError(mode)
        out = tmp.loc[idx].copy().rename(columns={'lab_value': name, 'lab_offset': f'{name}_offset_min', 'labname': f'{name}_labname'})
        return out[['patientunitstayid', name, f'{name}_offset_min', f'{name}_labname']]
    if strategy == 'first':
        ph = pick(ph_mask, 'ph', 'first')
        lact = pick(lact_mask, 'lactate', 'first')
        bun = pick(bun_mask, 'bun', 'first')
        urea_direct = pick(urea_direct_mask, 'urea_direct', 'first')
    else:
        ph = pick(ph_mask, 'ph', 'worst_min')
        lact = pick(lact_mask, 'lactate', 'worst_max')
        bun = pick(bun_mask, 'bun', 'worst_max')
        urea_direct = pick(urea_direct_mask, 'urea_direct', 'worst_max')
    out = pd.DataFrame({'patientunitstayid': sorted(ids)})
    out = out.merge(ph, on='patientunitstayid', how='left')
    out = out.merge(lact, on='patientunitstayid', how='left')
    out = out.merge(bun, on='patientunitstayid', how='left')
    out = out.merge(urea_direct, on='patientunitstayid', how='left')
    out['urea_from_bun'] = out['bun'] * 2.14
    out['urea'] = out['urea_from_bun'].where(out['urea_from_bun'].notna(), out['urea_direct'])
    out['urea_source'] = np.where(out['urea_from_bun'].notna(), 'BUN_mgdl_x2.14', np.where(out['urea_direct'].notna(), 'direct_urea', 'missing'))
    return out

def concat_text_columns(df: pd.DataFrame, exclude: set[str] | None=None) -> pd.Series:
    exclude = exclude or set()
    cols = [c for c in df.columns if c not in exclude]
    if not cols:
        return pd.Series([''] * len(df), index=df.index)
    tmp = df[cols].copy().fillna('')
    for c in cols:
        tmp[c] = tmp[c].map(lambda x: '' if pd.isna(x) else str(x))
    return tmp.apply(lambda row: ' '.join([str(v) for v in row.values if str(v) != '']), axis=1)

def extract_resp_support(resp: pd.DataFrame | None, treatment: pd.DataFrame | None, ids: set[int]) -> pd.DataFrame:
    out = pd.DataFrame({'patientunitstayid': sorted(ids), 'resp_support': 0})
    support_ids = set()
    if treatment is not None and 'patientunitstayid' in treatment.columns:
        text = clean_text(concat_text_columns(treatment, exclude=set()))
        mask = text.str.contains('non-invasive|noninvasive|nippv|nppv|\\bniv\\b|cpap|bipap|bi-pap|mechanical ventilation|ventilator|intubat|endotracheal|tracheostomy', regex=True, na=False)
        support_ids |= set(treatment.loc[mask, 'patientunitstayid'].dropna().astype(int))
    if resp is not None and 'patientunitstayid' in resp.columns:
        text = clean_text(concat_text_columns(resp, exclude=set()))
        mask = text.str.contains('\\bcpap\\b|bipap|bi-pap|non.?invasive|nippv|nppv|\\bniv\\b|ventilator|endotracheal|\\bett\\b|trach|intubat|tidal volume|assist control|simv|prvc', regex=True, na=False)
        support_ids |= set(resp.loc[mask, 'patientunitstayid'].dropna().astype(int))
    out['resp_support'] = out['patientunitstayid'].isin(support_ids).astype(int)
    return out

def select_first_admission(df: pd.DataFrame) -> pd.DataFrame:
    if 'uniquepid' not in df.columns:
        return df.copy()
    sort_col = None
    for c in ['hospitaladmitoffset', 'unitadmitoffset', 'unitadmittime24', 'patientunitstayid']:
        if c in df.columns:
            sort_col = c
            break
    out = df.copy()
    out['_sort'] = pd.to_numeric(out[sort_col], errors='coerce') if sort_col else pd.to_numeric(out['patientunitstayid'], errors='coerce')
    out = out.sort_values(['uniquepid', '_sort', 'patientunitstayid']).groupby('uniquepid', as_index=False).first()
    return out.drop(columns=['_sort'], errors='ignore')

@dataclass
class ArgsLike:
    eicu_dir: Path
    medians: Path | None
    output: Path
    lab_strategy: str
    lab_start_min: int
    lab_end_min: int
    exclude_competing: bool
    ph_low: float
    ph_high: float
    lactate_low: float
    lactate_high: float
    urea_low: float
    urea_high: float
    cv_splits: int
    cv_repeats: int

def build_dataset(args: ArgsLike) -> tuple[pd.DataFrame, dict[str, float]]:
    eicu = Path(args.eicu_dir)
    patient = norm_cols(read_csv(find_file(eicu, 'patient')))
    diagnosis = norm_cols(read_csv(find_file(eicu, 'diagnosis')))
    lab = norm_cols(read_csv(find_file(eicu, 'lab')))
    past = norm_cols(read_csv(find_file(eicu, 'pastHistory')))
    treatment_path = find_file(eicu, 'treatment', required=False)
    resp_path = find_file(eicu, 'respiratoryCharting', required=False)
    treatment = norm_cols(read_csv(treatment_path)) if treatment_path else None
    resp = norm_cols(read_csv(resp_path)) if resp_path else None
    medians = load_medians(args.medians)
    ids = extract_aecopd_ids(diagnosis, exclude_competing=args.exclude_competing)
    events = extract_events(diagnosis, ids)
    history = extract_history(past, ids)
    labs = extract_labs(lab, ids, args.lab_start_min, args.lab_end_min, args.lab_strategy)
    resp_support = extract_resp_support(resp, treatment, ids)
    patient = patient[patient['patientunitstayid'].isin(ids)].copy()
    patient['age'] = prepare_age(patient)
    base_cols = ['patientunitstayid', 'age']
    for c in ['uniquepid', 'gender', 'hospitalid', 'hospitaladmitoffset', 'unitadmitoffset', 'unitadmittime24']:
        if c in patient.columns:
            base_cols.append(c)
    df = patient[base_cols].merge(history, on='patientunitstayid', how='left')
    df = df.merge(labs, on='patientunitstayid', how='left')
    df = df.merge(resp_support, on='patientunitstayid', how='left')
    df = df.merge(events, on='patientunitstayid', how='left')
    for c in ['history_hf', 'history_af', 'resp_support', 'cv_event']:
        df[c] = df[c].fillna(0).astype(int)
    for c in ['mi_event', 'pe_event', 'pulmonary_edema_event', 'acute_arrhythmia_event', 'hf_decompensation_event']:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(int)
    for v in ['age', 'ph', 'urea', 'lactate']:
        df[v] = pd.to_numeric(df[v], errors='coerce')
    for v in ['ph', 'urea', 'lactate']:
        df[f'{v}_missing'] = df[v].isna().astype(int)
    plausible = (df['age'].isna() | df['age'].between(18, 110)) & (df['ph'].isna() | df['ph'].between(6.5, 8.0)) & (df['lactate'].isna() | df['lactate'].between(0, 40)) & (df['urea'].isna() | df['urea'].between(0, 500))
    df = df[plausible].copy()
    df = select_first_admission(df)
    for v in CORE:
        df[v] = pd.to_numeric(df[v], errors='coerce').fillna(medians[v])
    df['ph'] = df['ph'].clip(args.ph_low, args.ph_high)
    df['lactate'] = df['lactate'].clip(args.lactate_low, args.lactate_high)
    df['urea'] = df['urea'].clip(args.urea_low, args.urea_high)
    df['lactate_log1p'] = np.log1p(df['lactate'].clip(lower=0))
    df['urea_log1p'] = np.log1p(df['urea'].clip(lower=0))
    df['ph_acidemia'] = np.maximum(0.0, 7.35 - df['ph'])
    df['hf_lactate'] = df['history_hf'] * df['lactate']
    df['af_lactate'] = df['history_af'] * df['lactate']
    df['ph_lactate'] = df['ph_acidemia'] * df['lactate']
    return (df, medians)

def cv_predict_model(df: pd.DataFrame, y: np.ndarray, features: list[str], splits: int, repeats: int, penalty: str='l2') -> np.ndarray:
    X = df[features].copy()
    for c in features:
        X[c] = pd.to_numeric(X[c], errors='coerce')
    if penalty == 'none':
        try:
            clf = LogisticRegression(penalty=None, solver='lbfgs', max_iter=5000, random_state=RANDOM_STATE)
        except TypeError:
            clf = LogisticRegression(penalty='none', solver='lbfgs', max_iter=5000, random_state=RANDOM_STATE)
    else:
        clf = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=5000, random_state=RANDOM_STATE)
    pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('model', clf)])
    cv = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats, random_state=RANDOM_STATE)
    pred_sum = np.zeros(len(df), dtype=float)
    pred_n = np.zeros(len(df), dtype=float)
    for tr, te in cv.split(X, y):
        pipe.fit(X.iloc[tr], y[tr])
        pred_sum[te] += pipe.predict_proba(X.iloc[te])[:, 1]
        pred_n[te] += 1
    return pred_sum / np.maximum(pred_n, 1)

def calibration_stats(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-08, 1 - 1e-08)
    y = np.asarray(y_true, dtype=int)
    logit_p = np.log(p / (1 - p))
    X = sm.add_constant(logit_p, has_constant='add')
    model = sm.Logit(y, X).fit(disp=False, maxiter=1000)
    vals = np.asarray(model.params, dtype=float)
    return (float(vals[0]), float(vals[1]))

def wilson_ci(k: np.ndarray, n: np.ndarray, z: float=1.96) -> tuple[np.ndarray, np.ndarray]:
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    p = np.divide(k, n, out=np.zeros_like(k), where=n > 0)
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (np.clip(center - half, 0, 1), np.clip(center + half, 0, 1))

def net_benefit(y_true: np.ndarray, pred_prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(pred_prob).astype(float)
    n = len(y)
    out = np.full(len(thresholds), np.nan)
    for i, pt in enumerate(thresholds):
        pred_pos = p >= pt
        tp = np.sum((pred_pos == 1) & (y == 1))
        fp = np.sum((pred_pos == 1) & (y == 0))
        out[i] = tp / n - fp / n * (pt / (1 - pt))
    return out

def treat_all_net_benefit(y_true: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    prevalence = np.mean(np.asarray(y_true).astype(int))
    return prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))

def smooth_ma(y: np.ndarray, window: int=5) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    if window <= 1 or len(arr) < window:
        return arr.copy()
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode='edge')
    return np.convolve(padded, kernel, mode='valid')

def bootstrap_net_benefit_ci(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray, n_boot: int=300, seed: int=84) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = np.zeros((n_boot, len(thresholds)), dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b, :] = net_benefit(y[idx], p[idx], thresholds)
    return (boots.mean(axis=0), np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0))

def build_external_median_sensitivity_df(df_primary: pd.DataFrame, medians: dict[str, float], args: ArgsLike) -> tuple[pd.DataFrame, pd.DataFrame]:
    lab_vars = ['ph', 'urea', 'lactate']
    sensitivity_df = df_primary.copy()
    rows = []
    for v in lab_vars:
        observed_mask = pd.to_numeric(df_primary[f'{v}_missing'], errors='coerce').fillna(1).astype(int).eq(0)
        observed_values = pd.to_numeric(df_primary.loc[observed_mask, v], errors='coerce').dropna()
        if observed_values.empty:
            raise ValueError(f'No observed 6-hour values available for {v}.')
        external_median = float(observed_values.median())
        missing_mask = pd.to_numeric(sensitivity_df[f'{v}_missing'], errors='coerce').fillna(1).astype(int).eq(1)
        sensitivity_df.loc[missing_mask, v] = external_median
        rows.append({'variable': v, 'frozen_derivation_median': float(medians[v]), 'eicu_observed_6h_median': external_median, 'eicu_observed_6h_mean': float(observed_values.mean()), 'observed_n': int(observed_mask.sum()), 'observed_pct': float(observed_mask.mean() * 100), 'missing_n': int((~observed_mask).sum()), 'missing_pct': float((~observed_mask).mean() * 100)})
    sensitivity_df['ph'] = sensitivity_df['ph'].clip(args.ph_low, args.ph_high)
    sensitivity_df['lactate'] = sensitivity_df['lactate'].clip(args.lactate_low, args.lactate_high)
    sensitivity_df['urea'] = sensitivity_df['urea'].clip(args.urea_low, args.urea_high)
    return (sensitivity_df, pd.DataFrame(rows))

def summarize_selected_model(y: np.ndarray, p: np.ndarray, analysis_label: str) -> dict:
    auc = float(roc_auc_score(y, p))
    auc_lo, auc_hi = bootstrap_auc_ci(y, p, seed=105)
    ap = float(average_precision_score(y, p))
    ap_lo, ap_hi = bootstrap_metric_ci(y, p, average_precision_score, seed=205)
    brier = float(brier_score_loss(y, p))
    cal_i, cal_s = calibration_stats(y, p)
    return {'analysis': analysis_label, 'model': 'Updated 6 predictors + missingness + respiratory support', 'n': int(len(y)), 'events': int(np.sum(y)), 'event_rate': float(np.mean(y)), 'auc': auc, 'auc_ci_low': auc_lo, 'auc_ci_high': auc_hi, 'average_precision': ap, 'average_precision_ci_low': ap_lo, 'average_precision_ci_high': ap_hi, 'brier_score': brier, 'calibration_intercept': float(cal_i), 'calibration_slope': float(cal_s)}

def quartile_table(y, p, strategy):
    d = pd.DataFrame({'y': np.asarray(y, dtype=int), 'pred': np.asarray(p, dtype=float)}).dropna()
    try:
        d['quartile'] = pd.qcut(d['pred'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
    except ValueError:
        d['quartile'] = pd.cut(d['pred'], bins=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], include_lowest=True)
    g = d.groupby('quartile', observed=False).agg(n=('y', 'size'), events=('y', 'sum'), event_rate=('y', 'mean'), mean_pred=('pred', 'mean')).reset_index()
    g = g[g['n'] > 0].copy()
    g['ci_low'], g['ci_high'] = wilson_ci(g['events'].values, g['n'].values)
    g.insert(0, 'strategy', strategy)
    return g

def make_roc_plot(y, p1, p2, performance, outpath):
    fpr1, tpr1, _ = roc_curve(y, p1)
    fpr2, tpr2, _ = roc_curve(y, p2)
    r1, r2 = (performance.iloc[0], performance.iloc[1])
    plt.figure(figsize=(6.8, 6.0))
    plt.plot(fpr1, tpr1, linewidth=2.4, label=f"Frozen derivation medians (AUC {r1['auc']:.3f}, 95% CI {r1['auc_ci_low']:.3f}–{r1['auc_ci_high']:.3f})")
    plt.plot(fpr2, tpr2, linewidth=2.4, label=f"eICU 6-h medians (AUC {r2['auc']:.3f}, 95% CI {r2['auc_ci_low']:.3f}–{r2['auc_ci_high']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1.2)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('eICU external-validation imputation sensitivity: ROC curves')
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()

def lowess_curve(y, p, seed):
    p = np.clip(np.asarray(p, dtype=float), 1e-06, 1 - 1e-06)
    y = np.asarray(y, dtype=int)
    d = pd.DataFrame({'y': y, 'pred': p}).sort_values('pred').reset_index(drop=True)
    xmax = min(0.9, float(np.quantile(d['pred'], 0.98)))
    dp = d[d['pred'] <= xmax].copy()
    if len(dp) < 30:
        dp = d.copy()
        xmax = min(0.9, float(d['pred'].max()))
    curve = lowess(dp['y'], dp['pred'], frac=0.72, it=0, return_sorted=True)
    grid = np.linspace(float(dp['pred'].min()), float(dp['pred'].max()), 160)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(180):
        idx = rng.integers(0, len(dp), size=len(dp))
        b = dp.iloc[idx].sort_values('pred')
        lo_b = lowess(b['y'], b['pred'], frac=0.72, it=0, return_sorted=True)
        xb, yb = (np.asarray(lo_b[:, 0]), np.asarray(lo_b[:, 1]))
        keep = np.unique(xb, return_index=True)[1]
        xu, yu = (xb[np.sort(keep)], yb[np.sort(keep)])
        if len(xu) >= 2:
            boots.append(np.interp(grid, xu, yu, left=yu[0], right=yu[-1]))
    if len(boots) > 20:
        boots = np.asarray(boots)
        lo_ci = np.clip(np.percentile(boots, 2.5, axis=0), 0, 0.95)
        hi_ci = np.clip(np.percentile(boots, 97.5, axis=0), 0, 0.95)
    else:
        lo_ci = hi_ci = None
    ci, cs = calibration_stats(y, p)
    return (curve, grid, lo_ci, hi_ci, xmax, ci, cs)

def make_calibration_plot(y, p1, p2, outpath, table_out):
    c1, g1, l1, h1, x1, ci1, cs1 = lowess_curve(y, p1, 1702)
    c2, g2, l2, h2, x2, ci2, cs2 = lowess_curve(y, p2, 1703)
    xmax = max(x1, x2)
    plt.figure(figsize=(6.8, 6.0))
    plt.plot([0, xmax], [0, xmax], linestyle='--', linewidth=1.2, label='Ideal calibration')
    if l1 is not None:
        plt.fill_between(g1, l1, h1, alpha=0.1)
    if l2 is not None:
        plt.fill_between(g2, l2, h2, alpha=0.1)
    plt.plot(c1[:, 0], c1[:, 1], linewidth=2.4, label='Frozen derivation medians')
    plt.plot(c2[:, 0], c2[:, 1], linewidth=2.4, label='eICU 6-h medians')
    ymax = max(0.75, min(0.95, max(float(np.nanmax(c1[:, 1])), float(np.nanmax(c2[:, 1]))) + 0.08))
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)
    plt.xlabel('Predicted probability')
    plt.ylabel('Observed event rate')
    plt.title('eICU external-validation imputation sensitivity: calibration')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()
    tbl = pd.DataFrame([{'strategy': 'Frozen derivation-cohort medians', 'calibration_intercept': ci1, 'calibration_slope': cs1, 'x_max_plot': x1}, {'strategy': 'eICU observed 6-hour medians', 'calibration_intercept': ci2, 'calibration_slope': cs2, 'x_max_plot': x2}])
    tbl.to_csv(table_out, index=False)
    return tbl

def make_dca_plot(y, p1, p2, outpath, table_out):
    thresholds = np.arange(0.1, 0.5 + 1e-09, 0.01)
    m1, l1, h1 = bootstrap_net_benefit_ci(y, p1, thresholds, n_boot=300, seed=84)
    m2, l2, h2 = bootstrap_net_benefit_ci(y, p2, thresholds, n_boot=300, seed=85)
    all_nb = treat_all_net_benefit(y, thresholds)
    none_nb = np.zeros_like(thresholds)
    plt.figure(figsize=(8, 6.8))
    plt.fill_between(thresholds, smooth_ma(l1, 5), smooth_ma(h1, 5), alpha=0.1)
    plt.fill_between(thresholds, smooth_ma(l2, 5), smooth_ma(h2, 5), alpha=0.1)
    plt.plot(thresholds, smooth_ma(m1, 5), linewidth=2.4, label='Frozen derivation medians')
    plt.plot(thresholds, smooth_ma(m2, 5), linewidth=2.4, label='eICU 6-h medians')
    plt.plot(thresholds, smooth_ma(all_nb, 5), linewidth=2.0, label='Treat all')
    plt.axhline(0, linestyle='--', linewidth=1.5, label='Treat none')
    plt.xlim(0.1, 0.5)
    ymin = min(np.nanmin(smooth_ma(l1, 5)), np.nanmin(smooth_ma(l2, 5)), np.nanmin(smooth_ma(all_nb, 5)), -0.02)
    ymax = max(np.nanmax(smooth_ma(h1, 5)), np.nanmax(smooth_ma(h2, 5)), np.nanmax(smooth_ma(all_nb, 5)), 0.02)
    plt.ylim(ymin - 0.03, ymax + 0.03)
    plt.xlabel('Threshold probability')
    plt.ylabel('Net benefit')
    plt.title('eICU external-validation imputation sensitivity: decision curve')
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()
    tbl = pd.DataFrame({'threshold': thresholds, 'frozen_derivation_medians_nb': m1, 'frozen_derivation_medians_nb_low': l1, 'frozen_derivation_medians_nb_high': h1, 'eicu_6h_medians_nb': m2, 'eicu_6h_medians_nb_low': l2, 'eicu_6h_medians_nb_high': h2, 'treat_all_nb': all_nb, 'treat_none_nb': none_nb})
    tbl.to_csv(table_out, index=False)
    return tbl

def make_quartile_plot(q1, q2, outpath):
    labels = ['Q1', 'Q2', 'Q3', 'Q4']
    x = np.arange(4)
    off = 0.07
    a = q1.copy()
    a['quartile'] = a['quartile'].astype(str)
    a = a.set_index('quartile').reindex(labels)
    b = q2.copy()
    b['quartile'] = b['quartile'].astype(str)
    b = b.set_index('quartile').reindex(labels)
    plt.figure(figsize=(6.8, 5.6))
    plt.errorbar(x - off, a['event_rate'], yerr=[a['event_rate'] - a['ci_low'], a['ci_high'] - a['event_rate']], fmt='o', capsize=4, linewidth=1.5, markersize=7, label='Frozen derivation medians')
    plt.errorbar(x + off, b['event_rate'], yerr=[b['event_rate'] - b['ci_low'], b['ci_high'] - b['event_rate']], fmt='o', capsize=4, linewidth=1.5, markersize=7, label='eICU 6-h medians')
    plt.plot(x - off, a['event_rate'], linewidth=1.5)
    plt.plot(x + off, b['event_rate'], linewidth=1.5)
    plt.xticks(x, labels)
    plt.ylabel('Observed event rate')
    plt.xlabel('Predicted-risk quartile')
    plt.title('eICU external-validation imputation sensitivity: risk quartiles')
    plt.ylim(0, max(0.4, float(max(a['ci_high'].max(), b['ci_high'].max())) + 0.05))
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()

def make_composite_figure(roc_path, dca_path, cal_path, quart_path, outpath):
    imgs = [Image.open(p).convert('RGB') for p in [roc_path, dca_path, cal_path, quart_path]]
    w = max((i.width for i in imgs))
    h = max((i.height for i in imgs))
    canvas = Image.new('RGB', (2 * w, 2 * h), 'white')
    draw = ImageDraw.Draw(canvas)
    for idx, img in enumerate(imgs):
        x = idx % 2 * w
        y = idx // 2 * h
        canvas.paste(img.resize((w, h)), (x, y))
        draw.text((x + 10, y + 10), ['(A)', '(B)', '(C)', '(D)'][idx], fill='black')
    canvas.save(outpath)

def main() -> None:
    parser = argparse.ArgumentParser(description='eICU 6-hour laboratory-imputation sensitivity analysis for the selected model.')
    parser.add_argument('--eicu-dir', required=True)
    parser.add_argument('--medians', default=None)
    parser.add_argument('--output', required=True)
    parser.add_argument('--lab-strategy', choices=['first', 'worst'], default='first')
    parser.add_argument('--lab-start-min', type=int, default=0)
    parser.add_argument('--lab-end-min', type=int, default=360)
    parser.add_argument('--exclude-competing', action='store_true')
    parser.add_argument('--ph-low', type=float, default=6.8)
    parser.add_argument('--ph-high', type=float, default=7.8)
    parser.add_argument('--lactate-low', type=float, default=0.0)
    parser.add_argument('--lactate-high', type=float, default=20.0)
    parser.add_argument('--urea-low', type=float, default=0.0)
    parser.add_argument('--urea-high', type=float, default=300.0)
    parser.add_argument('--cv-splits', type=int, default=5)
    parser.add_argument('--cv-repeats', type=int, default=10)
    parsed = parser.parse_args()
    args = ArgsLike(eicu_dir=Path(parsed.eicu_dir), medians=Path(parsed.medians) if parsed.medians else None, output=Path(parsed.output), lab_strategy=parsed.lab_strategy, lab_start_min=parsed.lab_start_min, lab_end_min=parsed.lab_end_min, exclude_competing=parsed.exclude_competing, ph_low=parsed.ph_low, ph_high=parsed.ph_high, lactate_low=parsed.lactate_low, lactate_high=parsed.lactate_high, urea_low=parsed.urea_low, urea_high=parsed.urea_high, cv_splits=parsed.cv_splits, cv_repeats=parsed.cv_repeats)
    set_plot_style()
    args.output.mkdir(parents=True, exist_ok=True)
    primary_data, development_medians = build_dataset(args)
    if primary_data.empty:
        raise ValueError('No eligible patients.')
    y = primary_data['cv_event'].astype(int).to_numpy()
    if np.unique(y).size < 2:
        raise ValueError('The outcome contains only one class.')
    sensitivity_data, imputation_values = build_external_median_sensitivity_df(primary_data, development_medians, args)
    features = CORE + MISS + RESP
    primary_predictions = cv_predict_model(primary_data, y, features, args.cv_splits, args.cv_repeats)
    sensitivity_predictions = cv_predict_model(sensitivity_data, y, features, args.cv_splits, args.cv_repeats)
    performance = pd.DataFrame([summarize_selected_model(y, primary_predictions, 'Frozen derivation-cohort medians'), summarize_selected_model(y, sensitivity_predictions, 'eICU observed 6-hour medians')])
    performance.to_csv(args.output / 'imputation_sensitivity_performance_6h.csv', index=False)
    availability = pd.DataFrame([{'lab_window': f'{args.lab_start_min}-{args.lab_end_min} min', 'lab_strategy': args.lab_strategy, 'lab_selection_rule': 'earliest available value within window' if args.lab_strategy == 'first' else 'worst value within window', 'n': int(len(primary_data)), 'events': int(y.sum()), 'event_rate': float(y.mean()), 'ph_available_n': int((primary_data['ph_missing'] == 0).sum()), 'ph_available_pct': float((primary_data['ph_missing'] == 0).mean() * 100), 'urea_available_n': int((primary_data['urea_missing'] == 0).sum()), 'urea_available_pct': float((primary_data['urea_missing'] == 0).mean() * 100), 'lactate_available_n': int((primary_data['lactate_missing'] == 0).sum()), 'lactate_available_pct': float((primary_data['lactate_missing'] == 0).mean() * 100), 'all_three_labs_available_n': int(((primary_data['ph_missing'] == 0) & (primary_data['urea_missing'] == 0) & (primary_data['lactate_missing'] == 0)).sum()), 'all_three_labs_available_pct': float(((primary_data['ph_missing'] == 0) & (primary_data['urea_missing'] == 0) & (primary_data['lactate_missing'] == 0)).mean() * 100)}])
    availability.to_csv(args.output / 'imputation_sensitivity_lab_availability_6h.csv', index=False)
    imputation_values.to_csv(args.output / 'imputation_sensitivity_values_6h.csv', index=False)
    primary_quartiles = quartile_table(y, primary_predictions, 'Frozen derivation-cohort medians')
    sensitivity_quartiles = quartile_table(y, sensitivity_predictions, 'eICU observed 6-hour medians')
    quartiles = pd.concat([primary_quartiles, sensitivity_quartiles], ignore_index=True)
    quartiles.to_csv(args.output / 'imputation_sensitivity_quartiles_6h.csv', index=False)
    roc_path = args.output / 'figure_imputation_sensitivity_roc_6h.png'
    dca_path = args.output / 'figure_imputation_sensitivity_dca_6h.png'
    calibration_path = args.output / 'figure_imputation_sensitivity_calibration_6h.png'
    quartile_path = args.output / 'figure_imputation_sensitivity_quartiles_6h.png'
    make_roc_plot(y, primary_predictions, sensitivity_predictions, performance, roc_path)
    calibration_table = make_calibration_plot(y, primary_predictions, sensitivity_predictions, calibration_path, args.output / 'imputation_sensitivity_calibration_6h.csv')
    decision_curve_table = make_dca_plot(y, primary_predictions, sensitivity_predictions, dca_path, args.output / 'imputation_sensitivity_decision_curve_6h.csv')
    make_quartile_plot(primary_quartiles, sensitivity_quartiles, quartile_path)
    composite_path = args.output / 'eICU_Imputation_Sensitivity_Composite_6h.png'
    make_composite_figure(roc_path, dca_path, calibration_path, quartile_path, composite_path)
    patient_level = primary_data.copy()
    patient_level['primary_predicted_probability'] = primary_predictions
    patient_level['sensitivity_predicted_probability'] = sensitivity_predictions
    patient_level.to_csv(args.output / 'imputation_sensitivity_patient_level_6h.csv', index=False)
    workbook_path = args.output / 'eICU_imputation_sensitivity_6h.xlsx'
    with pd.ExcelWriter(workbook_path, engine='openpyxl', mode='w') as writer:
        performance.to_excel(writer, sheet_name='Performance', index=False)
        imputation_values.to_excel(writer, sheet_name='Imputation values', index=False)
        availability.to_excel(writer, sheet_name='Lab availability', index=False)
        calibration_table.to_excel(writer, sheet_name='Calibration', index=False)
        quartiles.to_excel(writer, sheet_name='Risk quartiles', index=False)
        decision_curve_table.to_excel(writer, sheet_name='Decision curve', index=False)
    summary = {'selected_model': 'Updated 6 predictors + missingness + respiratory support', 'selected_model_features': features, 'n': int(len(primary_data)), 'events': int(y.sum()), 'laboratory_window': '0-360 minutes', 'primary_strategy': 'Frozen derivation-cohort median imputation', 'sensitivity_strategy': 'Observed eICU 6-hour median imputation', 'cv_splits': args.cv_splits, 'cv_repeats': args.cv_repeats, 'random_state': RANDOM_STATE, 'development_medians': {variable: float(development_medians[variable]) for variable in ['ph', 'urea', 'lactate']}, 'eicu_observed_6h_medians': {str(row['variable']): float(row['eicu_observed_6h_median']) for _, row in imputation_values.iterrows()}, 'outputs': {'composite_figure': str(composite_path), 'excel_workbook': str(workbook_path)}}
    (args.output / 'imputation_sensitivity_summary_6h.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Done.')
    print(performance.to_string(index=False))
    print(f'Outputs: {args.output}')
if __name__ == '__main__':
    main()
