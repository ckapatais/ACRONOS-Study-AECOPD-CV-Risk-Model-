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
MODEL_FEATURES = ["age", "history_hf", "history_af", "ph", "urea", "lactate"]
LAB_COLUMNS = {"ph": "ph_6h", "urea": "urea_6h", "lactate": "lactate_6h"}


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


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def inv_logit(linear_predictor: np.ndarray) -> np.ndarray:
    values = np.asarray(linear_predictor, dtype=float)
    values = np.clip(values, -50, 50)
    return 1.0 / (1.0 + np.exp(-values))


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

    if not estimates:
        return np.nan, np.nan

    return (
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5)),
    )


def calibration_stats(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    probabilities = np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)
    outcomes = np.asarray(y, dtype=int)
    linear_predictor = np.log(probabilities / (1 - probabilities))
    design = sm.add_constant(linear_predictor, has_constant="add")
    model = sm.Logit(outcomes, design).fit(disp=False, maxiter=1000)
    params = np.asarray(model.params, dtype=float)
    return float(params[0]), float(params[1])


def load_metadata(input_path: Path, metadata_path: str | None) -> tuple[dict, Path]:
    path = Path(metadata_path) if metadata_path else input_path.parent / "frozen_prediction_generation_metadata.json"
    if not path.exists():
        raise FileNotFoundError(
            "Model metadata file not found. Provide its location with --metadata."
        )

    metadata = json.loads(path.read_text(encoding="utf-8"))

    required = {"intercept", "betas", "medians"}
    missing = required.difference(metadata)
    if missing:
        raise ValueError(f"Model metadata is missing required fields: {sorted(missing)}")

    for feature in MODEL_FEATURES:
        if feature not in metadata["betas"]:
            raise ValueError(f"Missing coefficient for {feature}")
        if feature not in metadata["medians"]:
            raise ValueError(f"Missing development median for {feature}")

    return metadata, path


def validate_columns(df: pd.DataFrame) -> None:
    required = [
        "cv_event",
        "age",
        "history_hf",
        "history_af",
        "ph_6h",
        "urea_6h",
        "lactate_6h",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")


def observed_lab_medians(df: pd.DataFrame) -> dict[str, float]:
    medians = {}

    for feature, column in LAB_COLUMNS.items():
        values = pd.to_numeric(df[column], errors="coerce")
        median = values.median(skipna=True)
        if pd.isna(median):
            raise ValueError(f"No observed values are available for {column}")
        medians[feature] = float(median)

    return medians


def prepare_predictors(
    df: pd.DataFrame,
    development_medians: dict[str, float],
    laboratory_medians: dict[str, float],
) -> pd.DataFrame:
    predictors = pd.DataFrame(index=df.index)

    predictors["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(
        development_medians["age"]
    )
    predictors["history_hf"] = pd.to_numeric(
        df["history_hf"], errors="coerce"
    ).fillna(development_medians["history_hf"])
    predictors["history_af"] = pd.to_numeric(
        df["history_af"], errors="coerce"
    ).fillna(development_medians["history_af"])

    for feature, column in LAB_COLUMNS.items():
        predictors[feature] = pd.to_numeric(
            df[column], errors="coerce"
        ).fillna(laboratory_medians[feature])

    return predictors


def predict(
    predictors: pd.DataFrame,
    intercept: float,
    betas: dict[str, float],
) -> np.ndarray:
    linear_predictor = np.full(len(predictors), float(intercept), dtype=float)

    for feature in MODEL_FEATURES:
        linear_predictor += float(betas[feature]) * predictors[feature].to_numpy(
            dtype=float
        )

    return inv_logit(linear_predictor)


def performance_row(
    strategy: str,
    y: np.ndarray,
    p: np.ndarray,
    seed: int,
) -> dict:
    auc = float(roc_auc_score(y, p))
    auc_low, auc_high = bootstrap_auc_ci(y, p, n_boot=2000, seed=seed)
    calibration_intercept, calibration_slope = calibration_stats(y, p)

    return {
        "strategy": strategy,
        "n": int(len(y)),
        "events": int(y.sum()),
        "event_rate": float(np.mean(y)),
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
    proportions = np.divide(
        events,
        totals,
        out=np.zeros_like(events),
        where=totals > 0,
    )
    denominator = 1 + z**2 / totals
    center = (proportions + z**2 / (2 * totals)) / denominator
    half_width = (
        z
        * np.sqrt(
            proportions * (1 - proportions) / totals
            + z**2 / (4 * totals**2)
        )
        / denominator
    )
    return (
        np.clip(center - half_width, 0, 1),
        np.clip(center + half_width, 0, 1),
    )


def net_benefit(
    y: np.ndarray,
    p: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    outcomes = np.asarray(y, dtype=int)
    probabilities = np.asarray(p, dtype=float)
    n = len(outcomes)
    values = np.full(len(thresholds), np.nan)

    for i, threshold in enumerate(thresholds):
        positive = probabilities >= threshold
        true_positive = np.sum(positive & (outcomes == 1))
        false_positive = np.sum(positive & (outcomes == 0))
        values[i] = (
            true_positive / n
            - false_positive / n * threshold / (1 - threshold)
        )

    return values


def treat_all_net_benefit(
    y: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    prevalence = np.mean(np.asarray(y, dtype=int))
    return prevalence - (1 - prevalence) * thresholds / (1 - thresholds)


def smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    values = np.asarray(values, dtype=float)

    if window <= 1 or len(values) < window:
        return values.copy()

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
    n = len(y)
    estimates = np.zeros((n_boot, len(thresholds)), dtype=float)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        estimates[i] = net_benefit(y[idx], p[idx], thresholds)

    return (
        estimates.mean(axis=0),
        np.percentile(estimates, 2.5, axis=0),
        np.percentile(estimates, 97.5, axis=0),
    )


def quartile_table(
    y: np.ndarray,
    p: np.ndarray,
    strategy: str,
) -> pd.DataFrame:
    data = pd.DataFrame({"outcome": y, "predicted_risk": p}).dropna()

    try:
        data["quartile"] = pd.qcut(
            data["predicted_risk"],
            q=4,
            labels=["Q1", "Q2", "Q3", "Q4"],
            duplicates="drop",
        )
    except ValueError:
        data["quartile"] = pd.cut(
            data["predicted_risk"],
            bins=4,
            labels=["Q1", "Q2", "Q3", "Q4"],
            include_lowest=True,
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


def make_roc_plot(
    y: np.ndarray,
    primary: np.ndarray,
    sensitivity: np.ndarray,
    performance: pd.DataFrame,
    output_path: Path,
) -> None:
    primary_fpr, primary_tpr, _ = roc_curve(y, primary)
    sensitivity_fpr, sensitivity_tpr, _ = roc_curve(y, sensitivity)

    primary_row = performance.iloc[0]
    sensitivity_row = performance.iloc[1]

    plt.figure(figsize=(6.8, 6.0))
    plt.plot(
        primary_fpr,
        primary_tpr,
        linewidth=2.2,
        label=(
            f"Frozen derivation medians "
            f"(AUC {primary_row['auc']:.3f}, "
            f"95% CI {primary_row['auc_ci_low']:.3f}–"
            f"{primary_row['auc_ci_high']:.3f})"
        ),
    )
    plt.plot(
        sensitivity_fpr,
        sensitivity_tpr,
        linewidth=2.2,
        label=(
            f"MIMIC-IV 6-h medians "
            f"(AUC {sensitivity_row['auc']:.3f}, "
            f"95% CI {sensitivity_row['auc_ci_low']:.3f}–"
            f"{sensitivity_row['auc_ci_high']:.3f})"
        ),
    )
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("MIMIC-IV 6-hour imputation sensitivity: ROC curves")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def lowess_curve_with_ci(
    y: np.ndarray,
    p: np.ndarray,
    seed: int,
    frac: float = 0.72,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, float]:
    probabilities = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    outcomes = np.asarray(y, dtype=int)

    data = (
        pd.DataFrame({"outcome": outcomes, "predicted_risk": probabilities})
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
        frac=frac,
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
        bootstrap_curve = lowess(
            sample["outcome"],
            sample["predicted_risk"],
            frac=frac,
            it=0,
            return_sorted=True,
        )

        x_values = np.asarray(bootstrap_curve[:, 0], dtype=float)
        y_values = np.asarray(bootstrap_curve[:, 1], dtype=float)
        unique_idx = np.unique(x_values, return_index=True)[1]
        unique_idx = np.sort(unique_idx)
        x_values = x_values[unique_idx]
        y_values = y_values[unique_idx]

        if len(x_values) >= 2:
            bootstrap_curves.append(
                np.interp(
                    grid,
                    x_values,
                    y_values,
                    left=y_values[0],
                    right=y_values[-1],
                )
            )

    if len(bootstrap_curves) > 20:
        bootstrap_curves = np.asarray(bootstrap_curves)
        ci_low = np.clip(
            np.percentile(bootstrap_curves, 2.5, axis=0),
            0,
            0.95,
        )
        ci_high = np.clip(
            np.percentile(bootstrap_curves, 97.5, axis=0),
            0,
            0.95,
        )
    else:
        ci_low = None
        ci_high = None

    return curve, grid, ci_low, ci_high, x_max


def make_calibration_plot(
    y: np.ndarray,
    primary: np.ndarray,
    sensitivity: np.ndarray,
    output_path: Path,
    table_path: Path,
) -> pd.DataFrame:
    primary_curve, primary_grid, primary_low, primary_high, primary_xmax = (
        lowess_curve_with_ci(y, primary, seed=1702)
    )
    sensitivity_curve, sensitivity_grid, sensitivity_low, sensitivity_high, sensitivity_xmax = (
        lowess_curve_with_ci(y, sensitivity, seed=1703)
    )

    x_max = max(primary_xmax, sensitivity_xmax)
    primary_intercept, primary_slope = calibration_stats(y, primary)
    sensitivity_intercept, sensitivity_slope = calibration_stats(y, sensitivity)

    plt.figure(figsize=(6.2, 5.6))
    plt.plot(
        [0, x_max],
        [0, x_max],
        linestyle="--",
        linewidth=1.2,
        label="Ideal calibration",
    )

    if primary_low is not None:
        plt.fill_between(
            primary_grid,
            primary_low,
            primary_high,
            alpha=0.10,
        )

    if sensitivity_low is not None:
        plt.fill_between(
            sensitivity_grid,
            sensitivity_low,
            sensitivity_high,
            alpha=0.10,
        )

    plt.plot(
        primary_curve[:, 0],
        primary_curve[:, 1],
        linewidth=2.4,
        label="Frozen derivation medians",
    )
    plt.plot(
        sensitivity_curve[:, 0],
        sensitivity_curve[:, 1],
        linewidth=2.4,
        label="MIMIC-IV 6-h medians",
    )

    y_max = max(
        0.75,
        min(
            0.95,
            max(
                float(np.nanmax(primary_curve[:, 1])),
                float(np.nanmax(sensitivity_curve[:, 1])),
            )
            + 0.08,
        ),
    )

    plt.xlim(0, x_max)
    plt.ylim(0, y_max)
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed event rate")
    plt.title("External validation calibration plot")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    table = pd.DataFrame(
        [
            {
                "strategy": "Frozen derivation-cohort medians",
                "calibration_intercept": primary_intercept,
                "calibration_slope": primary_slope,
                "x_max_plot": primary_xmax,
            },
            {
                "strategy": "MIMIC-IV observed 6h medians",
                "calibration_intercept": sensitivity_intercept,
                "calibration_slope": sensitivity_slope,
                "x_max_plot": sensitivity_xmax,
            },
        ]
    )
    table.to_csv(table_path, index=False)
    return table


def make_dca_plot(
    y: np.ndarray,
    primary: np.ndarray,
    sensitivity: np.ndarray,
    output_path: Path,
    table_path: Path,
) -> pd.DataFrame:
    thresholds = np.arange(0.10, 0.50 + 1e-9, 0.01)

    primary_mean, primary_low, primary_high = bootstrap_net_benefit_ci(
        y,
        primary,
        thresholds,
        n_boot=300,
        seed=84,
    )
    sensitivity_mean, sensitivity_low, sensitivity_high = bootstrap_net_benefit_ci(
        y,
        sensitivity,
        thresholds,
        n_boot=300,
        seed=85,
    )

    treat_all = treat_all_net_benefit(y, thresholds)
    treat_none = np.zeros_like(thresholds)

    plt.figure(figsize=(8, 6.8))
    plt.fill_between(
        thresholds,
        smooth(primary_low, 5),
        smooth(primary_high, 5),
        alpha=0.10,
    )
    plt.fill_between(
        thresholds,
        smooth(sensitivity_low, 5),
        smooth(sensitivity_high, 5),
        alpha=0.10,
    )
    plt.plot(
        thresholds,
        smooth(primary_mean, 5),
        linewidth=2.4,
        label="Frozen derivation medians",
    )
    plt.plot(
        thresholds,
        smooth(sensitivity_mean, 5),
        linewidth=2.4,
        label="MIMIC-IV 6-h medians",
    )
    plt.plot(
        thresholds,
        smooth(treat_all, 5),
        linewidth=2.0,
        label="Treat all",
    )
    plt.axhline(
        0,
        linestyle="--",
        linewidth=1.5,
        label="Treat none",
    )

    y_min = min(
        np.nanmin(smooth(primary_low, 5)),
        np.nanmin(smooth(sensitivity_low, 5)),
        np.nanmin(smooth(treat_all, 5)),
        -0.02,
    )
    y_max = max(
        np.nanmax(smooth(primary_high, 5)),
        np.nanmax(smooth(sensitivity_high, 5)),
        np.nanmax(smooth(treat_all, 5)),
        0.02,
    )

    plt.xlim(0.10, 0.50)
    plt.ylim(y_min - 0.03, y_max + 0.03)
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title("External validation decision curve analysis")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    table = pd.DataFrame(
        {
            "threshold": thresholds,
            "frozen_derivation_medians_nb": primary_mean,
            "frozen_derivation_medians_nb_low": primary_low,
            "frozen_derivation_medians_nb_high": primary_high,
            "mimic_iv_6h_medians_nb": sensitivity_mean,
            "mimic_iv_6h_medians_nb_low": sensitivity_low,
            "mimic_iv_6h_medians_nb_high": sensitivity_high,
            "treat_all_nb": treat_all,
            "treat_none_nb": treat_none,
        }
    )
    table.to_csv(table_path, index=False)
    return table


def make_quartile_plot(
    primary_table: pd.DataFrame,
    sensitivity_table: pd.DataFrame,
    output_path: Path,
) -> None:
    labels = ["Q1", "Q2", "Q3", "Q4"]
    x = np.arange(len(labels))
    offset = 0.07

    primary = primary_table.copy()
    primary["quartile"] = primary["quartile"].astype(str)
    primary = primary.set_index("quartile").reindex(labels)

    sensitivity = sensitivity_table.copy()
    sensitivity["quartile"] = sensitivity["quartile"].astype(str)
    sensitivity = sensitivity.set_index("quartile").reindex(labels)

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
        label="Frozen derivation medians",
    )
    plt.errorbar(
        x + offset,
        sensitivity["event_rate"],
        yerr=[
            sensitivity["event_rate"] - sensitivity["ci_low"],
            sensitivity["ci_high"] - sensitivity["event_rate"],
        ],
        fmt="o",
        capsize=4,
        linewidth=1.5,
        markersize=7,
        label="MIMIC-IV 6-h medians",
    )
    plt.plot(x - offset, primary["event_rate"], linewidth=1.5)
    plt.plot(x + offset, sensitivity["event_rate"], linewidth=1.5)
    plt.xticks(x, labels)
    plt.ylabel("Observed event rate")
    plt.xlabel("Predicted-risk quartile")
    plt.title("Observed event rate across predicted-risk quartiles")
    plt.ylim(
        0,
        max(
            0.40,
            float(
                max(
                    primary["ci_high"].max(),
                    sensitivity["ci_high"].max(),
                )
            )
            + 0.05,
        ),
    )
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def make_composite_figure(
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
    labels = ["(A)", "(B)", "(C)", "(D)"]
    draw = ImageDraw.Draw(canvas)

    for i, image in enumerate(images):
        x = (i % 2) * width
        y = (i // 2) * height
        canvas.paste(image.resize((width, height)), (x, y))
        draw.text((x + 10, y + 10), labels[i], fill="black")

    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MIMIC-IV 6-hour laboratory-imputation sensitivity analysis."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Patient-level MIMIC-IV 6-hour validation dataset.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory for tables and figures.",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Model metadata JSON file.",
    )
    args = parser.parse_args()

    set_plot_style()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = read_table(input_path)
    validate_columns(data)
    metadata, metadata_path = load_metadata(input_path, args.metadata)

    y = pd.to_numeric(data["cv_event"], errors="coerce").astype(int).to_numpy()

    development_medians = {
        key: float(value)
        for key, value in metadata["medians"].items()
    }
    betas = {
        key: float(value)
        for key, value in metadata["betas"].items()
    }
    intercept = float(metadata["intercept"])
    mimic_medians = observed_lab_medians(data)

    primary_predictors = prepare_predictors(
        data,
        development_medians,
        {
            "ph": development_medians["ph"],
            "urea": development_medians["urea"],
            "lactate": development_medians["lactate"],
        },
    )
    sensitivity_predictors = prepare_predictors(
        data,
        development_medians,
        mimic_medians,
    )

    primary_predictions = predict(
        primary_predictors,
        intercept,
        betas,
    )
    sensitivity_predictions = predict(
        sensitivity_predictors,
        intercept,
        betas,
    )

    performance = pd.DataFrame(
        [
            performance_row(
                "Frozen derivation-cohort medians",
                y,
                primary_predictions,
                seed=6101,
            ),
            performance_row(
                "MIMIC-IV observed 6h medians",
                y,
                sensitivity_predictions,
                seed=6101,
            ),
        ]
    )
    performance.to_csv(
        output_dir / "imputation_sensitivity_performance_6h.csv",
        index=False,
    )

    value_rows = []
    availability_rows = []

    for feature, column in LAB_COLUMNS.items():
        values = pd.to_numeric(data[column], errors="coerce")
        observed_n = int(values.notna().sum())
        missing_n = int(values.isna().sum())

        value_rows.append(
            {
                "variable": feature,
                "development_median": development_medians[feature],
                "mimic_iv_observed_6h_median": mimic_medians[feature],
                "mimic_iv_observed_6h_mean": float(values.mean(skipna=True)),
                "observed_n": observed_n,
                "observed_pct": float(values.notna().mean() * 100),
                "missing_n": missing_n,
                "missing_pct": float(values.isna().mean() * 100),
            }
        )

        availability_rows.append(
            {
                "variable": feature,
                "available_n": observed_n,
                "available_pct": float(values.notna().mean() * 100),
                "missing_n": missing_n,
                "missing_pct": float(values.isna().mean() * 100),
            }
        )

    imputation_values = pd.DataFrame(value_rows)
    imputation_values.to_csv(
        output_dir / "imputation_sensitivity_values_6h.csv",
        index=False,
    )

    complete_labs = pd.DataFrame(
        {
            feature: pd.to_numeric(data[column], errors="coerce").notna()
            for feature, column in LAB_COLUMNS.items()
        }
    ).all(axis=1)

    availability_rows.append(
        {
            "variable": "all_three_6h_labs_complete",
            "available_n": int(complete_labs.sum()),
            "available_pct": float(complete_labs.mean() * 100),
            "missing_n": int((~complete_labs).sum()),
            "missing_pct": float((~complete_labs).mean() * 100),
        }
    )

    availability = pd.DataFrame(availability_rows)
    availability.to_csv(
        output_dir / "imputation_sensitivity_lab_availability_6h.csv",
        index=False,
    )

    primary_quartiles = quartile_table(
        y,
        primary_predictions,
        "Frozen derivation-cohort medians",
    )
    sensitivity_quartiles = quartile_table(
        y,
        sensitivity_predictions,
        "MIMIC-IV observed 6h medians",
    )
    quartiles = pd.concat(
        [primary_quartiles, sensitivity_quartiles],
        ignore_index=True,
    )
    quartiles.to_csv(
        output_dir / "imputation_sensitivity_quartiles_6h.csv",
        index=False,
    )

    roc_path = output_dir / "figure_imputation_sensitivity_roc_6h.png"
    dca_path = output_dir / "figure_imputation_sensitivity_dca_6h.png"
    calibration_path = output_dir / "figure_imputation_sensitivity_calibration_6h.png"
    quartile_path = output_dir / "figure_imputation_sensitivity_quartiles_6h.png"

    make_roc_plot(
        y,
        primary_predictions,
        sensitivity_predictions,
        performance,
        roc_path,
    )

    decision_curve = make_dca_plot(
        y,
        primary_predictions,
        sensitivity_predictions,
        dca_path,
        output_dir / "imputation_sensitivity_decision_curve_6h.csv",
    )

    calibration = make_calibration_plot(
        y,
        primary_predictions,
        sensitivity_predictions,
        calibration_path,
        output_dir / "imputation_sensitivity_calibration_6h.csv",
    )

    make_quartile_plot(
        primary_quartiles,
        sensitivity_quartiles,
        quartile_path,
    )

    composite_path = output_dir / "MIMIC_IV_Imputation_Sensitivity_Composite_6h.png"
    make_composite_figure(
        roc_path,
        dca_path,
        calibration_path,
        quartile_path,
        composite_path,
    )

    patient_level = data.copy()
    patient_level["primary_predicted_probability"] = primary_predictions
    patient_level["sensitivity_predicted_probability"] = sensitivity_predictions
    patient_level.to_csv(
        output_dir / "imputation_sensitivity_patient_level_6h.csv",
        index=False,
    )

    workbook_path = output_dir / "MIMIC_IV_imputation_sensitivity_6h.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl", mode="w") as writer:
        performance.to_excel(writer, sheet_name="Performance", index=False)
        imputation_values.to_excel(writer, sheet_name="Imputation values", index=False)
        availability.to_excel(writer, sheet_name="Lab availability", index=False)
        calibration.to_excel(writer, sheet_name="Calibration", index=False)
        quartiles.to_excel(writer, sheet_name="Risk quartiles", index=False)
        decision_curve.to_excel(writer, sheet_name="Decision curve", index=False)

    summary = {
        "input_file": str(input_path),
        "metadata_file": str(metadata_path),
        "n": int(len(data)),
        "events": int(y.sum()),
        "laboratory_window": "6 hours",
        "primary_strategy": "Development-cohort median imputation",
        "sensitivity_strategy": "MIMIC-IV observed 6-hour median imputation",
        "model_features": MODEL_FEATURES,
        "outputs": {
            "composite_figure": str(composite_path),
            "excel_workbook": str(workbook_path),
        },
    }

    (
        output_dir / "imputation_sensitivity_summary_6h.json"
    ).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Done.")
    print(performance.to_string(index=False))
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
