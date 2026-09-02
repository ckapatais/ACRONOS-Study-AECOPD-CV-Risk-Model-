from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from PIL import Image, ImageDraw
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from statsmodels.nonparametric.smoothers_lowess import lowess


RANDOM_STATE = 42
CHUNK_SIZE = 1_000_000
PH_ITEMID = 50820
SPECIMEN_ITEMID = 52033
MODEL_FEATURES = ["age", "history_hf", "history_af", "ph", "urea", "lactate"]

EMBEDDED_COEFFICIENTS = {
    "const": 46.95019433422684,
    "age": 0.0056297841639299,
    "history_hf": 0.5535675156229428,
    "history_af": 1.2434884123076857,
    "ph": -6.746131471731943,
    "urea": 0.0069608750607488,
    "lactate": 0.1190962972054364,
}

DEVELOPMENT_MEDIANS = {
    "age": 72.0,
    "history_hf": 0.0,
    "history_af": 0.0,
    "ph": 7.39,
    "urea": 34.05,
    "lactate": 1.2,
}


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 9.5,
            "axes.linewidth": 0.9,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "legend.frameon": False,
        }
    )


def find_table(folder: Path, name: str) -> Path:
    for suffix in [".csv.gz", ".csv", ".CSV.GZ", ".CSV"]:
        candidate = folder / f"{name}{suffix}"
        if candidate.exists():
            return candidate
    for suffix in [".csv.gz", ".csv", ".CSV.GZ", ".CSV"]:
        matches = list(folder.rglob(f"{name}{suffix}"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find {name}.csv or {name}.csv.gz under {folder}")


def read_patient_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)


def normalize_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    aliases = {
        "HF_history": "history_hf",
        "AF_history": "history_af",
        "ph": "ph_6h",
        "urea": "urea_6h",
        "lactate": "lactate_6h",
    }
    for old, new in aliases.items():
        if new not in data.columns and old in data.columns:
            data[new] = data[old]
    required = [
        "subject_id",
        "hadm_id",
        "cv_event",
        "age",
        "history_hf",
        "history_af",
        "ph_6h",
        "urea_6h",
        "lactate_6h",
    ]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")
    data["subject_id"] = pd.to_numeric(data["subject_id"], errors="raise").astype(int)
    data["hadm_id"] = pd.to_numeric(data["hadm_id"], errors="raise").astype(int)
    data["cv_event"] = pd.to_numeric(data["cv_event"], errors="raise").astype(int)
    return data


def load_coefficients(path: Path | None) -> dict[str, float]:
    if path is None:
        return EMBEDDED_COEFFICIENTS.copy()

    table = pd.read_csv(path)
    table.columns = [str(column).strip() for column in table.columns]

    variable_column = next(
        (column for column in ["variable", "term", "feature", "predictor"] if column in table.columns),
        None,
    )
    beta_column = next(
        (column for column in ["beta", "coef", "coefficient", "estimate"] if column in table.columns),
        None,
    )

    if variable_column is None or beta_column is None:
        raise ValueError("Coefficient file must contain variable and beta columns.")

    values = {}
    for _, row in table.iterrows():
        variable = str(row[variable_column]).strip()
        beta = pd.to_numeric(row[beta_column], errors="coerce")
        if pd.isna(beta):
            continue
        if variable.lower() in {"const", "intercept", "(intercept)"}:
            values["const"] = float(beta)
        elif variable in MODEL_FEATURES:
            values[variable] = float(beta)

    required = ["const"] + MODEL_FEATURES
    missing = [feature for feature in required if feature not in values]
    if missing:
        raise ValueError(f"Coefficient file is missing: {missing}")

    return values


def load_admission_times(mimic_root: Path, cohort: pd.DataFrame) -> pd.DataFrame:
    admissions_path = find_table(mimic_root / "hosp", "admissions")
    admissions = pd.read_csv(
        admissions_path,
        usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
        low_memory=False,
    )
    admissions["subject_id"] = pd.to_numeric(admissions["subject_id"], errors="coerce").astype("Int64")
    admissions["hadm_id"] = pd.to_numeric(admissions["hadm_id"], errors="coerce").astype("Int64")
    admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"], errors="coerce")

    keys = cohort[["subject_id", "hadm_id"]].drop_duplicates()
    merged = keys.merge(admissions, on=["subject_id", "hadm_id"], how="left", validate="one_to_one")

    if merged["admittime"].isna().any():
        n_missing = int(merged["admittime"].isna().sum())
        raise ValueError(f"{n_missing} cohort admissions could not be linked to MIMIC-IV admissions.")

    merged["window_end"] = merged["admittime"] + pd.Timedelta(hours=6)
    return merged


def extract_arterial_ph(
    mimic_root: Path,
    admission_times: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labevents_path = find_table(mimic_root / "hosp", "labevents")
    hadm_ids = set(admission_times["hadm_id"].astype(int))
    pieces = []

    usecols = [
        "subject_id",
        "hadm_id",
        "specimen_id",
        "itemid",
        "charttime",
        "value",
        "valuenum",
    ]

    for chunk in pd.read_csv(
        labevents_path,
        usecols=usecols,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        chunk["hadm_id"] = pd.to_numeric(chunk["hadm_id"], errors="coerce").astype("Int64")
        chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce").astype("Int64")
        chunk = chunk[
            chunk["hadm_id"].isin(hadm_ids)
            & chunk["itemid"].isin([PH_ITEMID, SPECIMEN_ITEMID])
        ].copy()
        if not chunk.empty:
            pieces.append(chunk)

    if not pieces:
        raise ValueError("No blood-gas pH/specimen records were found for the cohort.")

    blood_gas = pd.concat(pieces, ignore_index=True)
    blood_gas["specimen_id"] = pd.to_numeric(blood_gas["specimen_id"], errors="coerce").astype("Int64")
    blood_gas["charttime"] = pd.to_datetime(blood_gas["charttime"], errors="coerce")
    blood_gas["valuenum"] = pd.to_numeric(blood_gas["valuenum"], errors="coerce")

    specimen = blood_gas[blood_gas["itemid"].eq(SPECIMEN_ITEMID)].copy()
    specimen["specimen_label"] = specimen["value"].astype(str).str.strip()
    specimen = (
        specimen.dropna(subset=["specimen_id"])
        .sort_values(["specimen_id", "charttime"])
        .drop_duplicates("specimen_id", keep="first")
        [["specimen_id", "specimen_label"]]
    )

    ph = blood_gas[blood_gas["itemid"].eq(PH_ITEMID)].copy()
    ph = ph.dropna(subset=["hadm_id", "specimen_id", "charttime", "valuenum"])
    ph = ph[ph["valuenum"].between(6.5, 8.0)].copy()
    ph = ph.merge(specimen, on="specimen_id", how="left")
    ph["specimen_label_normalized"] = (
        ph["specimen_label"].fillna("UNSPECIFIED").astype(str).str.strip().str.upper()
    )

    time_columns = admission_times[
        ["subject_id", "hadm_id", "admittime", "window_end", "dischtime"]
    ].copy()
    ph = ph.merge(
        time_columns,
        on=["subject_id", "hadm_id"],
        how="inner",
    )
    ph = ph[
        ph["charttime"].ge(ph["admittime"])
        & ph["charttime"].le(ph["window_end"])
        & (ph["dischtime"].isna() | ph["charttime"].le(ph["dischtime"]))
    ].copy()

    distribution = (
        ph.groupby("specimen_label_normalized", dropna=False)
        .agg(
            pH_measurements=("valuenum", "size"),
            unique_admissions=("hadm_id", "nunique"),
        )
        .reset_index()
        .sort_values(["unique_admissions", "pH_measurements"], ascending=False)
    )

    arterial = ph[
        ph["specimen_label_normalized"].str.startswith("ART", na=False)
    ].copy()
    arterial = (
        arterial.sort_values(["hadm_id", "charttime", "specimen_id"])
        .drop_duplicates("hadm_id", keep="first")
        .rename(
            columns={
                "valuenum": "arterial_ph_6h",
                "charttime": "arterial_ph_6h_charttime",
                "specimen_id": "arterial_ph_6h_specimen_id",
                "specimen_label_normalized": "arterial_ph_6h_specimen",
            }
        )
    )

    arterial = arterial[
        [
            "subject_id",
            "hadm_id",
            "arterial_ph_6h",
            "arterial_ph_6h_charttime",
            "arterial_ph_6h_specimen_id",
            "arterial_ph_6h_specimen",
        ]
    ]

    return arterial, distribution


def inv_logit(linear_predictor: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(linear_predictor, dtype=float), -50, 50)
    return 1.0 / (1.0 + np.exp(-values))


def build_predictors(
    cohort: pd.DataFrame,
    ph_column: str,
) -> pd.DataFrame:
    predictors = pd.DataFrame(index=cohort.index)
    predictors["age"] = pd.to_numeric(cohort["age"], errors="coerce")
    predictors["history_hf"] = pd.to_numeric(cohort["history_hf"], errors="coerce")
    predictors["history_af"] = pd.to_numeric(cohort["history_af"], errors="coerce")
    predictors["ph"] = pd.to_numeric(cohort[ph_column], errors="coerce")
    predictors["urea"] = pd.to_numeric(cohort["urea_6h"], errors="coerce")
    predictors["lactate"] = pd.to_numeric(cohort["lactate_6h"], errors="coerce")

    for feature in MODEL_FEATURES:
        predictors[feature] = predictors[feature].fillna(DEVELOPMENT_MEDIANS[feature])

    return predictors


def predict(
    predictors: pd.DataFrame,
    coefficients: dict[str, float],
) -> np.ndarray:
    linear_predictor = np.full(len(predictors), coefficients["const"], dtype=float)
    for feature in MODEL_FEATURES:
        linear_predictor += (
            coefficients[feature]
            * predictors[feature].to_numpy(dtype=float)
        )
    return inv_logit(linear_predictor)


def bootstrap_auc_ci(
    y: np.ndarray,
    p: np.ndarray,
    n_boot: int = 2000,
    seed: int = RANDOM_STATE,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = []
    n = len(y)

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_boot = y[idx]
        p_boot = p[idx]
        if np.unique(y_boot).size < 2:
            continue
        estimates.append(roc_auc_score(y_boot, p_boot))

    return (
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5)),
    )


def calibration_stats(
    y: np.ndarray,
    p: np.ndarray,
) -> tuple[float, float]:
    probabilities = np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)
    outcomes = np.asarray(y, dtype=int)
    linear_predictor = np.log(probabilities / (1 - probabilities))
    design = sm.add_constant(linear_predictor, has_constant="add")
    model = sm.Logit(outcomes, design).fit(disp=False, maxiter=1000)
    params = np.asarray(model.params, dtype=float)
    return float(params[0]), float(params[1])


def performance_row(
    strategy: str,
    y: np.ndarray,
    p: np.ndarray,
    seed: int,
) -> dict:
    auc = float(roc_auc_score(y, p))
    auc_low, auc_high = bootstrap_auc_ci(y, p, seed=seed)
    calibration_intercept, calibration_slope = calibration_stats(y, p)
    return {
        "strategy": strategy,
        "n": int(len(y)),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        "auc": auc,
        "auc_ci_low": auc_low,
        "auc_ci_high": auc_high,
        "brier_score": float(brier_score_loss(y, p)),
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
    }


def wilson_ci(
    events: np.ndarray,
    totals: np.ndarray,
    z: float = 1.96,
) -> tuple[np.ndarray, np.ndarray]:
    events = np.asarray(events, dtype=float)
    totals = np.asarray(totals, dtype=float)
    proportions = np.divide(events, totals, out=np.zeros_like(events), where=totals > 0)
    denominator = 1 + z**2 / totals
    center = (proportions + z**2 / (2 * totals)) / denominator
    half = (
        z
        * np.sqrt(
            proportions * (1 - proportions) / totals
            + z**2 / (4 * totals**2)
        )
        / denominator
    )
    return np.clip(center - half, 0, 1), np.clip(center + half, 0, 1)


def quartile_table(
    y: np.ndarray,
    p: np.ndarray,
    strategy: str,
) -> pd.DataFrame:
    data = pd.DataFrame({"outcome": y, "predicted_risk": p})
    data["quartile"] = pd.qcut(
        data["predicted_risk"],
        q=4,
        labels=["Q1", "Q2", "Q3", "Q4"],
        duplicates="drop",
    )
    grouped = (
        data.groupby("quartile", observed=False)
        .agg(
            n=("outcome", "size"),
            events=("outcome", "sum"),
            event_rate=("outcome", "mean"),
            mean_predicted_risk=("predicted_risk", "mean"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["n"] > 0].copy()
    grouped["ci_low"], grouped["ci_high"] = wilson_ci(
        grouped["events"].to_numpy(),
        grouped["n"].to_numpy(),
    )
    grouped.insert(0, "strategy", strategy)
    return grouped


def net_benefit(
    y: np.ndarray,
    p: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    outcomes = np.asarray(y, dtype=int)
    probabilities = np.asarray(p, dtype=float)
    n = len(outcomes)
    result = np.zeros(len(thresholds), dtype=float)

    for i, threshold in enumerate(thresholds):
        positive = probabilities >= threshold
        true_positive = np.sum(positive & (outcomes == 1))
        false_positive = np.sum(positive & (outcomes == 0))
        result[i] = (
            true_positive / n
            - false_positive / n * threshold / (1 - threshold)
        )

    return result


def treat_all_net_benefit(
    y: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    prevalence = np.mean(np.asarray(y, dtype=int))
    return prevalence - (1 - prevalence) * thresholds / (1 - thresholds)


def smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def bootstrap_net_benefit_ci(
    y: np.ndarray,
    p: np.ndarray,
    thresholds: np.ndarray,
    n_boot: int = 300,
    seed: int = 84,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    estimates = np.zeros((n_boot, len(thresholds)), dtype=float)
    n = len(y)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        estimates[i] = net_benefit(y[idx], p[idx], thresholds)

    return (
        estimates.mean(axis=0),
        np.percentile(estimates, 2.5, axis=0),
        np.percentile(estimates, 97.5, axis=0),
    )


def lowess_curve_with_ci(
    y: np.ndarray,
    p: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, float]:
    probabilities = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    data = (
        pd.DataFrame({"outcome": y, "predicted_risk": probabilities})
        .sort_values("predicted_risk")
        .reset_index(drop=True)
    )

    x_max = min(0.90, float(np.quantile(data["predicted_risk"], 0.98)))
    plot_data = data[data["predicted_risk"] <= x_max].copy()

    if len(plot_data) < 30:
        plot_data = data.copy()
        x_max = min(0.90, float(data["predicted_risk"].max()))

    curve = lowess(
        plot_data["outcome"],
        plot_data["predicted_risk"],
        frac=0.72,
        it=0,
        return_sorted=True,
    )
    grid = np.linspace(
        float(plot_data["predicted_risk"].min()),
        float(plot_data["predicted_risk"].max()),
        160,
    )

    rng = np.random.default_rng(seed)
    bootstrap_curves = []

    for _ in range(180):
        idx = rng.integers(0, len(plot_data), size=len(plot_data))
        sample = plot_data.iloc[idx].sort_values("predicted_risk")
        estimate = lowess(
            sample["outcome"],
            sample["predicted_risk"],
            frac=0.72,
            it=0,
            return_sorted=True,
        )
        x_values = estimate[:, 0]
        y_values = estimate[:, 1]
        keep = np.unique(x_values, return_index=True)[1]
        x_values = x_values[np.sort(keep)]
        y_values = y_values[np.sort(keep)]
        if len(x_values) >= 2:
            bootstrap_curves.append(
                np.interp(grid, x_values, y_values, left=y_values[0], right=y_values[-1])
            )

    if len(bootstrap_curves) > 20:
        bootstrap_curves = np.asarray(bootstrap_curves)
        ci_low = np.clip(np.percentile(bootstrap_curves, 2.5, axis=0), 0, 0.95)
        ci_high = np.clip(np.percentile(bootstrap_curves, 97.5, axis=0), 0, 0.95)
    else:
        ci_low = None
        ci_high = None

    return curve, grid, ci_low, ci_high, x_max


def make_roc_plot(
    y: np.ndarray,
    primary: np.ndarray,
    arterial: np.ndarray,
    performance: pd.DataFrame,
    output_path: Path,
) -> None:
    fpr_primary, tpr_primary, _ = roc_curve(y, primary)
    fpr_arterial, tpr_arterial, _ = roc_curve(y, arterial)
    row_primary = performance.iloc[0]
    row_arterial = performance.iloc[1]

    plt.figure(figsize=(6.8, 6.0))
    plt.plot(
        fpr_primary,
        tpr_primary,
        linewidth=2.4,
        label=(
            f"Primary pH extraction "
            f"(AUC {row_primary['auc']:.3f}, 95% CI "
            f"{row_primary['auc_ci_low']:.3f}–{row_primary['auc_ci_high']:.3f})"
        ),
    )
    plt.plot(
        fpr_arterial,
        tpr_arterial,
        linewidth=2.4,
        label=(
            f"Confirmed arterial pH only "
            f"(AUC {row_arterial['auc']:.3f}, 95% CI "
            f"{row_arterial['auc_ci_low']:.3f}–{row_arterial['auc_ci_high']:.3f})"
        ),
    )
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("MIMIC-IV arterial-pH sensitivity: ROC curves")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def make_calibration_plot(
    y: np.ndarray,
    primary: np.ndarray,
    arterial: np.ndarray,
    output_path: Path,
    table_path: Path,
) -> pd.DataFrame:
    c1, g1, l1, h1, x1 = lowess_curve_with_ci(y, primary, 1702)
    c2, g2, l2, h2, x2 = lowess_curve_with_ci(y, arterial, 1703)
    x_max = max(x1, x2)

    plt.figure(figsize=(6.8, 6.0))
    plt.plot([0, x_max], [0, x_max], linestyle="--", linewidth=1.2, label="Ideal calibration")
    if l1 is not None:
        plt.fill_between(g1, l1, h1, alpha=0.10)
    if l2 is not None:
        plt.fill_between(g2, l2, h2, alpha=0.10)
    plt.plot(c1[:, 0], c1[:, 1], linewidth=2.4, label="Primary pH extraction")
    plt.plot(c2[:, 0], c2[:, 1], linewidth=2.4, label="Confirmed arterial pH only")
    plt.xlim(0, x_max)
    plt.ylim(
        0,
        max(
            0.75,
            min(
                0.95,
                max(float(c1[:, 1].max()), float(c2[:, 1].max())) + 0.08,
            ),
        ),
    )
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed event rate")
    plt.title("MIMIC-IV arterial-pH sensitivity: calibration")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    primary_intercept, primary_slope = calibration_stats(y, primary)
    arterial_intercept, arterial_slope = calibration_stats(y, arterial)
    table = pd.DataFrame(
        [
            {
                "strategy": "Primary pH extraction",
                "calibration_intercept": primary_intercept,
                "calibration_slope": primary_slope,
            },
            {
                "strategy": "Confirmed arterial pH only",
                "calibration_intercept": arterial_intercept,
                "calibration_slope": arterial_slope,
            },
        ]
    )
    table.to_csv(table_path, index=False)
    return table


def make_dca_plot(
    y: np.ndarray,
    primary: np.ndarray,
    arterial: np.ndarray,
    output_path: Path,
    table_path: Path,
) -> pd.DataFrame:
    thresholds = np.arange(0.10, 0.50 + 1e-9, 0.01)
    m1, l1, h1 = bootstrap_net_benefit_ci(y, primary, thresholds, seed=84)
    m2, l2, h2 = bootstrap_net_benefit_ci(y, arterial, thresholds, seed=85)
    treat_all = treat_all_net_benefit(y, thresholds)

    plt.figure(figsize=(8, 6.8))
    plt.fill_between(thresholds, smooth(l1), smooth(h1), alpha=0.10)
    plt.fill_between(thresholds, smooth(l2), smooth(h2), alpha=0.10)
    plt.plot(thresholds, smooth(m1), linewidth=2.4, label="Primary pH extraction")
    plt.plot(thresholds, smooth(m2), linewidth=2.4, label="Confirmed arterial pH only")
    plt.plot(thresholds, smooth(treat_all), linewidth=2.0, label="Treat all")
    plt.axhline(0, linestyle="--", linewidth=1.5, label="Treat none")
    plt.xlim(0.10, 0.50)
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title("MIMIC-IV arterial-pH sensitivity: decision curve")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    table = pd.DataFrame(
        {
            "threshold": thresholds,
            "primary_ph_nb": m1,
            "primary_ph_nb_low": l1,
            "primary_ph_nb_high": h1,
            "arterial_ph_nb": m2,
            "arterial_ph_nb_low": l2,
            "arterial_ph_nb_high": h2,
            "treat_all_nb": treat_all,
            "treat_none_nb": np.zeros_like(thresholds),
        }
    )
    table.to_csv(table_path, index=False)
    return table


def make_quartile_plot(
    primary_table: pd.DataFrame,
    arterial_table: pd.DataFrame,
    output_path: Path,
) -> None:
    labels = ["Q1", "Q2", "Q3", "Q4"]
    x = np.arange(4)
    offset = 0.07

    primary = primary_table.copy()
    primary["quartile"] = primary["quartile"].astype(str)
    primary = primary.set_index("quartile").reindex(labels)

    arterial = arterial_table.copy()
    arterial["quartile"] = arterial["quartile"].astype(str)
    arterial = arterial.set_index("quartile").reindex(labels)

    plt.figure(figsize=(6.8, 5.6))
    plt.errorbar(
        x - offset,
        primary["event_rate"],
        yerr=[
            primary["event_rate"] - primary["ci_low"],
            primary["ci_high"] - primary["event_rate"],
        ],
        fmt="o",
        capsize=4,
        linewidth=1.5,
        markersize=7,
        label="Primary pH extraction",
    )
    plt.errorbar(
        x + offset,
        arterial["event_rate"],
        yerr=[
            arterial["event_rate"] - arterial["ci_low"],
            arterial["ci_high"] - arterial["event_rate"],
        ],
        fmt="o",
        capsize=4,
        linewidth=1.5,
        markersize=7,
        label="Confirmed arterial pH only",
    )
    plt.plot(x - offset, primary["event_rate"], linewidth=1.5)
    plt.plot(x + offset, arterial["event_rate"], linewidth=1.5)
    plt.xticks(x, labels)
    plt.xlabel("Predicted-risk quartile")
    plt.ylabel("Observed event rate")
    plt.title("MIMIC-IV arterial-pH sensitivity: risk quartiles")
    plt.ylim(
        0,
        max(
            0.40,
            float(max(primary["ci_high"].max(), arterial["ci_high"].max())) + 0.05,
        ),
    )
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def make_composite(
    roc_path: Path,
    dca_path: Path,
    calibration_path: Path,
    quartile_path: Path,
    output_path: Path,
) -> None:
    images = [
        Image.open(path).convert("RGB")
        for path in [roc_path, dca_path, calibration_path, quartile_path]
    ]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (2 * width, 2 * height), "white")
    draw = ImageDraw.Draw(canvas)

    for index, image in enumerate(images):
        x = (index % 2) * width
        y = (index // 2) * height
        canvas.paste(image.resize((width, height)), (x, y))
        draw.text((x + 10, y + 10), ["(A)", "(B)", "(C)", "(D)"][index], fill="black")

    canvas.save(output_path)


def missingness_outcome_table(cohort: pd.DataFrame) -> pd.DataFrame:
    rows = []
    definitions = {
        "Any primary 6-hour pH available": pd.to_numeric(cohort["ph_6h"], errors="coerce").notna(),
        "Confirmed arterial 6-hour pH available": pd.to_numeric(
            cohort["arterial_ph_6h"], errors="coerce"
        ).notna(),
    }

    for definition, available in definitions.items():
        for status, mask in [("available", available), ("not available", ~available)]:
            subset = cohort.loc[mask, "cv_event"]
            rows.append(
                {
                    "pH_definition": definition,
                    "status": status,
                    "n": int(mask.sum()),
                    "events": int(subset.sum()),
                    "event_rate": float(subset.mean()) if len(subset) else np.nan,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MIMIC-IV 6-hour sensitivity analysis restricted to confirmed arterial pH measurements."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--mimic-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--coefficients", default=None)
    args = parser.parse_args()

    set_plot_style()

    input_path = Path(args.input)
    mimic_root = Path(args.mimic_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cohort = normalize_input_columns(read_patient_file(input_path))
    coefficients = load_coefficients(Path(args.coefficients) if args.coefficients else None)
    admission_times = load_admission_times(mimic_root, cohort)
    arterial_ph, specimen_distribution = extract_arterial_ph(mimic_root, admission_times)

    cohort = cohort.merge(
        arterial_ph,
        on=["subject_id", "hadm_id"],
        how="left",
        validate="one_to_one",
    )

    primary_predictors = build_predictors(cohort, "ph_6h")
    arterial_predictors = build_predictors(cohort, "arterial_ph_6h")

    primary_predictions = predict(primary_predictors, coefficients)
    arterial_predictions = predict(arterial_predictors, coefficients)
    y = cohort["cv_event"].to_numpy(dtype=int)

    performance = pd.DataFrame(
        [
            performance_row("Primary pH extraction", y, primary_predictions, 6101),
            performance_row("Confirmed arterial pH only", y, arterial_predictions, 6101),
        ]
    )
    performance.to_csv(
        output_dir / "arterial_ph_sensitivity_performance_6h.csv",
        index=False,
    )

    primary_available = pd.to_numeric(cohort["ph_6h"], errors="coerce").notna()
    arterial_available = pd.to_numeric(cohort["arterial_ph_6h"], errors="coerce").notna()

    availability = pd.DataFrame(
        [
            {
                "definition": "Primary 6-hour pH",
                "available_n": int(primary_available.sum()),
                "available_pct": float(primary_available.mean() * 100),
                "missing_n": int((~primary_available).sum()),
                "missing_pct": float((~primary_available).mean() * 100),
            },
            {
                "definition": "Confirmed arterial 6-hour pH",
                "available_n": int(arterial_available.sum()),
                "available_pct": float(arterial_available.mean() * 100),
                "missing_n": int((~arterial_available).sum()),
                "missing_pct": float((~arterial_available).mean() * 100),
            },
        ]
    )
    availability["confirmed_arterial_as_pct_of_primary_available"] = np.nan
    if primary_available.sum() > 0:
        availability.loc[
            availability["definition"].eq("Confirmed arterial 6-hour pH"),
            "confirmed_arterial_as_pct_of_primary_available",
        ] = float(arterial_available.sum() / primary_available.sum() * 100)

    availability.to_csv(
        output_dir / "arterial_ph_availability_6h.csv",
        index=False,
    )
    specimen_distribution.to_csv(
        output_dir / "arterial_ph_specimen_distribution_6h.csv",
        index=False,
    )

    missingness_outcomes = missingness_outcome_table(cohort)
    missingness_outcomes.to_csv(
        output_dir / "arterial_ph_missingness_outcome_6h.csv",
        index=False,
    )

    primary_quartiles = quartile_table(y, primary_predictions, "Primary pH extraction")
    arterial_quartiles = quartile_table(y, arterial_predictions, "Confirmed arterial pH only")
    quartiles = pd.concat([primary_quartiles, arterial_quartiles], ignore_index=True)
    quartiles.to_csv(
        output_dir / "arterial_ph_sensitivity_quartiles_6h.csv",
        index=False,
    )

    roc_path = output_dir / "figure_arterial_ph_sensitivity_roc_6h.png"
    dca_path = output_dir / "figure_arterial_ph_sensitivity_dca_6h.png"
    calibration_path = output_dir / "figure_arterial_ph_sensitivity_calibration_6h.png"
    quartile_path = output_dir / "figure_arterial_ph_sensitivity_quartiles_6h.png"

    make_roc_plot(
        y,
        primary_predictions,
        arterial_predictions,
        performance,
        roc_path,
    )
    calibration = make_calibration_plot(
        y,
        primary_predictions,
        arterial_predictions,
        calibration_path,
        output_dir / "arterial_ph_sensitivity_calibration_6h.csv",
    )
    decision_curve = make_dca_plot(
        y,
        primary_predictions,
        arterial_predictions,
        dca_path,
        output_dir / "arterial_ph_sensitivity_decision_curve_6h.csv",
    )
    make_quartile_plot(primary_quartiles, arterial_quartiles, quartile_path)

    composite_path = output_dir / "MIMIC_IV_Arterial_pH_Sensitivity_Composite_6h.png"
    make_composite(
        roc_path,
        dca_path,
        calibration_path,
        quartile_path,
        composite_path,
    )

    patient_level = cohort.copy()
    patient_level["primary_model_probability"] = primary_predictions
    patient_level["arterial_ph_model_probability"] = arterial_predictions
    patient_level["primary_ph_available"] = primary_available.astype(int)
    patient_level["confirmed_arterial_ph_available"] = arterial_available.astype(int)
    patient_level.to_csv(
        output_dir / "arterial_ph_sensitivity_patient_level_6h.csv",
        index=False,
    )

    workbook_path = output_dir / "MIMIC_IV_arterial_ph_sensitivity_6h.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl", mode="w") as writer:
        performance.to_excel(writer, sheet_name="Performance", index=False)
        availability.to_excel(writer, sheet_name="pH availability", index=False)
        specimen_distribution.to_excel(writer, sheet_name="Specimen distribution", index=False)
        missingness_outcomes.to_excel(writer, sheet_name="Missingness outcomes", index=False)
        calibration.to_excel(writer, sheet_name="Calibration", index=False)
        quartiles.to_excel(writer, sheet_name="Risk quartiles", index=False)
        decision_curve.to_excel(writer, sheet_name="Decision curve", index=False)

    summary = {
        "n": int(len(cohort)),
        "events": int(y.sum()),
        "laboratory_window": "6 hours",
        "primary_pH_definition": "Primary structured pH extraction from the existing validation dataset",
        "sensitivity_pH_definition": "Earliest pH within 6 hours linked by specimen_id to a specimen label beginning with ART",
        "pH_itemid": PH_ITEMID,
        "specimen_itemid": SPECIMEN_ITEMID,
        "development_pH_median": DEVELOPMENT_MEDIANS["ph"],
        "unchanged": [
            "patient cohort",
            "outcome definition",
            "age",
            "history of heart failure",
            "history of atrial fibrillation",
            "observed urea",
            "observed lactate",
            "model coefficients",
            "model intercept",
            "prediction equation",
            "6-hour laboratory window",
        ],
        "changed_element": "Observed pH is restricted to confirmed arterial specimens; otherwise pH is assigned the development median.",
        "figure_generation": "ROC, calibration, decision-curve, and risk-quartile analyses use the directly generated model probabilities for both pH strategies.",
        "outputs": {
            "performance": str(output_dir / "arterial_ph_sensitivity_performance_6h.csv"),
            "availability": str(output_dir / "arterial_ph_availability_6h.csv"),
            "specimen_distribution": str(output_dir / "arterial_ph_specimen_distribution_6h.csv"),
            "missingness_outcomes": str(output_dir / "arterial_ph_missingness_outcome_6h.csv"),
            "composite_figure": str(composite_path),
            "workbook": str(workbook_path),
        },
    }

    (output_dir / "arterial_ph_sensitivity_summary_6h.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Done.")
    print("")
    print("pH availability:")
    print(availability.to_string(index=False))
    print("")
    print("Performance based on original model probabilities:")
    print(performance.to_string(index=False))
    print("")
    print("Outcome rates by pH availability:")
    print(missingness_outcomes.to_string(index=False))
    print("")
    print(f"Composite figure: {composite_path}")
    print(f"Workbook: {workbook_path}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
