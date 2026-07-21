"""
src/preprocessing.py
=====================
Pipeline de préprocessing partagé par le notebook 02 (exploration pédagogique)
et réutilisable tel quel par le backend pour le feature engineering à l'inférence.

Étapes (dans l'ordre imposé par la spécification) :
    1. Imputation par interpolation temporelle des valeurs manquantes
    2. Clipping des outliers (méthode IQR)
    3. Correction de la dérive lente des capteurs concernés
    4. Jointure asof avec le labo en respectant le délai de disponibilité (zéro fuite)
    5. Feature engineering (lags, moyennes glissantes, deltas, encodage crude_type)
    6. Construction des séquences (fenêtres glissantes) pour rendements et fouling
    7. Split temporel strict 70/15/15 (sans shuffle) + StandardScaler fit sur train uniquement
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config_loader import load_config, resolve_path

HIDDEN_ENERGY_COLS = ["fouling_resistance", "is_cleaning_event",
                       "cleaning_needed_within_24h", "cleaning_needed_within_48h"]


# -----------------------------------------------------------------------------
# 1-3. Nettoyage de la table CDU (seule table bruitée volontairement)
# -----------------------------------------------------------------------------
def load_raw_tables(cfg: dict) -> dict[str, pd.DataFrame]:
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    cdu = pd.read_csv(raw_dir / "cdu_data.csv", parse_dates=["timestamp"]).set_index("timestamp")
    energy = pd.read_csv(raw_dir / "energy_data.csv", parse_dates=["timestamp"]).set_index("timestamp")
    cracker = pd.read_csv(raw_dir / "cracker_data.csv", parse_dates=["timestamp"]).set_index("timestamp")
    lab = pd.read_csv(raw_dir / "lab_data.csv", parse_dates=["sample_time", "result_time"])
    return {"cdu": cdu, "energy": energy, "cracker": cracker, "lab": lab}


def impute_missing_temporal(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Interpolation temporelle (linéaire dans le temps) puis ffill/bfill pour les bords."""
    df = df.copy()
    df[numeric_cols] = df[numeric_cols].interpolate(method="time", limit_direction="both")
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    return df


def clip_outliers_iqr(df: pd.DataFrame, numeric_cols: list[str], k: float = 3.0) -> pd.DataFrame:
    """Winsorisation : les valeurs hors [Q1-k*IQR, Q3+k*IQR] sont ramenées aux bornes."""
    df = df.copy()
    for col in numeric_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        df[col] = df[col].clip(lower=lo, upper=hi)
    return df


def correct_sensor_drift(df: pd.DataFrame, drift_sensors: list[str], window_hours: int = 1080) -> pd.DataFrame:
    """Corrige la dérive lente : on retire la composante basse fréquence (médiane glissante
    longue) et on la recentre sur le niveau moyen global, ce qui préserve la dynamique
    opératoire rapide (réelle) tout en supprimant la dérive de calibration lente (artefact)."""
    df = df.copy()
    for col in drift_sensors:
        if col not in df.columns:
            continue
        trend = df[col].rolling(window=window_hours, center=True, min_periods=1).median()
        df[col] = df[col] - trend + trend.mean()
    return df


def clean_cdu_table(cdu: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    numeric_cols = [c for c in cdu.columns if c != "crude_type"]
    cdu = impute_missing_temporal(cdu, numeric_cols)
    cdu = clip_outliers_iqr(cdu, numeric_cols, k=cfg["preprocessing"]["outlier_iqr_k"])
    cdu = correct_sensor_drift(cdu, cfg["data_generator"]["cdu"]["drift_sensors"])
    cdu["crude_type"] = cdu["crude_type"].ffill().bfill()
    return cdu


# -----------------------------------------------------------------------------
# 4. Jointure asof avec le labo (zéro fuite : seul un résultat déjà disponible est utilisé)
# -----------------------------------------------------------------------------
def merge_lab_asof(hourly_df: pd.DataFrame, lab_df: pd.DataFrame) -> pd.DataFrame:
    lab_cols = ["naphtha_final_boiling_point", "kerosene_flash_point", "gasoil_cetane_index",
                "residue_viscosity", "sulfur_content"]
    lab_sorted = lab_df.sort_values("result_time")
    left = hourly_df.reset_index().rename(columns={"index": "timestamp"})
    merged = pd.merge_asof(
        left.sort_values("timestamp"), lab_sorted[["result_time"] + lab_cols],
        left_on="timestamp", right_on="result_time", direction="backward",
    )
    merged = merged.drop(columns=["result_time"]).set_index("timestamp")
    merged[lab_cols] = merged[lab_cols].ffill().bfill()
    return merged


# -----------------------------------------------------------------------------
# 5. Feature engineering
# -----------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame, lag_cols: list[str], rolling_cols: list[str],
                       rolling_windows: list[int], delta_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in lag_cols:
        for lag in (1, 3, 6):
            df[f"{col}_lag{lag}h"] = df[col].shift(lag)
    for col in rolling_cols:
        for w in rolling_windows:
            df[f"{col}_roll_mean_{w}h"] = df[col].rolling(w, min_periods=1).mean()
            df[f"{col}_roll_std_{w}h"] = df[col].rolling(w, min_periods=1).std().fillna(0.0)
    for col in delta_cols:
        df[f"{col}_delta1h"] = df[col].diff(1).fillna(0.0)

    if "crude_type" in df.columns:
        dummies = pd.get_dummies(df["crude_type"], prefix="crude").astype(float)
        df = pd.concat([df.drop(columns=["crude_type"]), dummies], axis=1)

    df = df.ffill().bfill()
    return df


def build_master_table(cfg: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construit la table maîtresse fusionnée (CDU nettoyée + énergie non-cachée + labo)
    avec feature engineering. Retourne (master_df, hidden_df) où hidden_df contient les
    colonnes de vérité terrain jamais utilisées comme feature (fouling_resistance, labels)."""
    cfg = cfg or load_config()
    tables = load_raw_tables(cfg)

    cdu_clean = clean_cdu_table(tables["cdu"], cfg)
    energy = tables["energy"]
    hidden_df = energy[HIDDEN_ENERGY_COLS].copy()
    energy_visible = energy.drop(columns=HIDDEN_ENERGY_COLS)

    merged = cdu_clean.join(energy_visible, how="inner")
    merged = merge_lab_asof(merged, tables["lab"])

    yield_cols = ["naphtha_yield", "kerosene_yield", "gasoil_yield", "residue_yield"]
    lag_cols = ["feed_rate", "furnace_cot", "reflux_ratio", "column_top_temp"] + yield_cols
    rolling_cols = ["feed_rate", "furnace_cot", "reflux_ratio", "preheat_outlet_temp"] + yield_cols
    delta_cols = ["furnace_cot", "reflux_ratio", "feed_rate"]
    rolling_windows = cfg["preprocessing"]["rolling_windows"]

    master = engineer_features(merged, lag_cols, rolling_cols, rolling_windows, delta_cols)
    hidden_df = hidden_df.reindex(master.index)
    return master, hidden_df


# -----------------------------------------------------------------------------
# 6. Construction des séquences
# -----------------------------------------------------------------------------
def build_yield_sequences(master: pd.DataFrame, window: int) -> tuple[np.ndarray, np.ndarray, list[str], pd.DatetimeIndex]:
    """Fenêtre de `window` heures -> rendements (naphta/kérosène/gazole/résidu) à t+1."""
    yield_cols = ["naphtha_yield", "kerosene_yield", "gasoil_yield", "residue_yield"]
    feature_cols = [c for c in master.columns if c not in []]  # toutes les features (yields inclus en autoregressif)
    values = master[feature_cols].values.astype(np.float32)
    targets = master[yield_cols].values.astype(np.float32)

    n = len(master)
    X, y, ts = [], [], []
    for t in range(window, n - 1):
        X.append(values[t - window:t])
        y.append(targets[t + 1])
        ts.append(master.index[t + 1])
    return np.array(X), np.array(y), feature_cols, pd.DatetimeIndex(ts)


def build_fouling_sequences(master: pd.DataFrame, hidden_df: pd.DataFrame, window: int,
                             clean_period_hours: int) -> dict[str, np.ndarray]:
    """Fenêtre de `window` heures de capteurs visibles ; cible = reconstruction de la fenêtre
    elle-même (non supervisé). Fournit aussi labels 24h/48h et un masque "période propre"
    (peu après un nettoyage) pour restreindre l'entraînement des autoencodeurs."""
    feature_cols = [c for c in master.columns]
    values = master[feature_cols].values.astype(np.float32)

    fouling = hidden_df["fouling_resistance"].values
    is_cleaning = hidden_df["is_cleaning_event"].values.astype(bool)
    label24 = hidden_df["cleaning_needed_within_24h"].values.astype(np.int64)
    label48 = hidden_df["cleaning_needed_within_48h"].values.astype(np.int64)

    hours_since_cleaning = np.zeros(len(master), dtype=np.float64)
    counter = clean_period_hours * 10  # grand au départ (avant le 1er nettoyage connu)
    for i in range(len(master)):
        if is_cleaning[i]:
            counter = 0
        else:
            counter += 1
        hours_since_cleaning[i] = counter

    n = len(master)
    X, ts, y24, y48, fouling_true, is_clean = [], [], [], [], [], []
    for t in range(window, n):
        X.append(values[t - window:t])
        ts.append(master.index[t])
        y24.append(label24[t])
        y48.append(label48[t])
        fouling_true.append(fouling[t])
        is_clean.append(hours_since_cleaning[t] <= clean_period_hours)

    return {
        "X": np.array(X), "timestamps": pd.DatetimeIndex(ts), "feature_cols": feature_cols,
        "label24": np.array(y24), "label48": np.array(y48),
        "fouling_true": np.array(fouling_true, dtype=np.float32),
        "is_clean": np.array(is_clean, dtype=bool),
    }


# -----------------------------------------------------------------------------
# 7. Split temporel strict + scaling
# -----------------------------------------------------------------------------
def temporal_split_indices(n: int, ratios: dict[str, float]) -> tuple[slice, slice, slice]:
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n)


def fit_scale_sequences(X_train: np.ndarray, X_val: np.ndarray,
                         X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """StandardScaler ajusté uniquement sur train (aplati sur l'axe temps), appliqué aux 3 splits."""
    n_feat = X_train.shape[-1]
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_feat))

    def _transform(X):
        shape = X.shape
        return scaler.transform(X.reshape(-1, n_feat)).reshape(shape).astype(np.float32)

    return _transform(X_train), _transform(X_val), _transform(X_test), scaler


# -----------------------------------------------------------------------------
# Orchestration complète (appelée par le notebook 02 et par `python -m src.preprocessing`)
# -----------------------------------------------------------------------------
def run_preprocessing(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    master, hidden_df = build_master_table(cfg)

    processed_dir = resolve_path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    ratios = cfg["preprocessing"]["split_ratios"]

    # --- Dataset rendements ---
    window_y = cfg["preprocessing"]["yield_window_hours"]
    Xy, yy, feat_cols_y, ts_y = build_yield_sequences(master, window_y)
    tr, va, te = temporal_split_indices(len(Xy), ratios)
    Xy_train, Xy_val, Xy_test, scaler_y = fit_scale_sequences(Xy[tr], Xy[va], Xy[te])

    np.save(processed_dir / "yields_X_train.npy", Xy_train)
    np.save(processed_dir / "yields_X_val.npy", Xy_val)
    np.save(processed_dir / "yields_X_test.npy", Xy_test)
    np.save(processed_dir / "yields_y_train.npy", yy[tr])
    np.save(processed_dir / "yields_y_val.npy", yy[va])
    np.save(processed_dir / "yields_y_test.npy", yy[te])
    np.save(processed_dir / "yields_timestamps_test.npy", ts_y[te].values)
    joblib.dump(scaler_y, processed_dir / "yields_scaler_X.joblib")
    joblib.dump(feat_cols_y, processed_dir / "yields_feature_names.joblib")

    # --- Dataset fouling ---
    window_f = cfg["preprocessing"]["fouling_window_hours"]
    clean_hours = cfg["alerts"]["fouling_warning_days"] * 24 * 3  # fenêtre "propre" = peu après nettoyage
    fdata = build_fouling_sequences(master, hidden_df, window_f, clean_period_hours=clean_hours)
    trf, vaf, tef = temporal_split_indices(len(fdata["X"]), ratios)
    Xf_train, Xf_val, Xf_test, scaler_f = fit_scale_sequences(fdata["X"][trf], fdata["X"][vaf], fdata["X"][tef])

    np.save(processed_dir / "fouling_X_train.npy", Xf_train)
    np.save(processed_dir / "fouling_X_val.npy", Xf_val)
    np.save(processed_dir / "fouling_X_test.npy", Xf_test)
    for name, sl in [("train", trf), ("val", vaf), ("test", tef)]:
        np.save(processed_dir / f"fouling_label24_{name}.npy", fdata["label24"][sl])
        np.save(processed_dir / f"fouling_label48_{name}.npy", fdata["label48"][sl])
        np.save(processed_dir / f"fouling_true_{name}.npy", fdata["fouling_true"][sl])
        np.save(processed_dir / f"fouling_is_clean_{name}.npy", fdata["is_clean"][sl])
    np.save(processed_dir / "fouling_timestamps_test.npy", fdata["timestamps"][tef].values)
    joblib.dump(scaler_f, processed_dir / "fouling_scaler_X.joblib")
    joblib.dump(fdata["feature_cols"], processed_dir / "fouling_feature_names.joblib")

    print(f"[preprocessing] Rendements : X={Xy.shape}, y={yy.shape} -> "
          f"train={Xy_train.shape[0]} val={Xy_val.shape[0]} test={Xy_test.shape[0]}")
    print(f"[preprocessing] Fouling    : X={fdata['X'].shape} -> "
          f"train={Xf_train.shape[0]} val={Xf_val.shape[0]} test={Xf_test.shape[0]}")
    print(f"[preprocessing] Artefacts sauvegardés dans {processed_dir}")

    return {
        "master": master, "hidden_df": hidden_df,
        "yields": {"X_train": Xy_train, "X_val": Xy_val, "X_test": Xy_test,
                   "y_train": yy[tr], "y_val": yy[va], "y_test": yy[te], "scaler": scaler_y},
        "fouling": {"X_train": Xf_train, "X_val": Xf_val, "X_test": Xf_test, "scaler": scaler_f, **fdata},
    }


if __name__ == "__main__":
    run_preprocessing()
