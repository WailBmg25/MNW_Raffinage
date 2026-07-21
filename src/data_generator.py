"""
src/data_generator.py
======================
Génération de données 100% synthétiques pour le jumeau numérique CDU & Vapocraqueur.

Basé sur des bilans matière/énergie simplifiés (pas de téléchargement externe).
Produit 4 tables horaires sur 2 ans (2024-01-01 -> 2025-12-31) dans data/raw/ :
    - cdu_data.csv
    - energy_data.csv
    - cracker_data.csv
    - lab_data.csv

Exécution :
    python -m src.data_generator
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config_loader import load_config, resolve_path
from src.seed_utils import set_global_seed


# -----------------------------------------------------------------------------
# Processus stochastiques élémentaires
# -----------------------------------------------------------------------------
def ou_process(n: int, mean: float, sigma: float, theta: float, rng: np.random.Generator,
                x0: float | None = None) -> np.ndarray:
    """Processus d'Ornstein-Uhlenbeck : bruit corrélé qui revient lentement vers `mean`.
    Utilisé pour simuler des variables de procédé réalistes (pas de bruit blanc pur)."""
    x = np.empty(n, dtype=np.float64)
    x[0] = mean if x0 is None else x0
    noise = rng.normal(0.0, 1.0, size=n)
    for t in range(1, n):
        x[t] = x[t - 1] + theta * (mean - x[t - 1]) + sigma * noise[t]
    return x


def make_regime_schedule(n_hours: int, min_days: float, max_days: float, rng: np.random.Generator) -> np.ndarray:
    """Construit un vecteur d'indices de segment (change de valeur tous les min_days..max_days jours)."""
    schedule = np.zeros(n_hours, dtype=int)
    t = 0
    seg_id = 0
    while t < n_hours:
        duration_h = int(rng.uniform(min_days, max_days) * 24)
        end = min(t + duration_h, n_hours)
        schedule[t:end] = seg_id
        seg_id += 1
        t = end
    return schedule


def smooth_transitions(values_by_hour: np.ndarray, schedule: np.ndarray, transition_hours: int) -> np.ndarray:
    """Lisse les transitions abruptes entre segments sur `transition_hours` heures (interpolation linéaire)."""
    out = values_by_hour.copy().astype(np.float64)
    change_points = np.where(np.diff(schedule) != 0)[0] + 1
    n = len(values_by_hour)
    half = transition_hours // 2
    for cp in change_points:
        lo = max(0, cp - half)
        hi = min(n, cp + half)
        if hi <= lo:
            continue
        before_val = values_by_hour[lo]
        after_val = values_by_hour[min(hi, n - 1)]
        ramp = np.linspace(0.0, 1.0, hi - lo)
        out[lo:hi] = before_val * (1 - ramp) + after_val * ramp
    return out


def inject_missing_and_outliers(df: pd.DataFrame, cols: list[str], missing_rate: float,
                                  outlier_rate: float, outlier_magnitude: float,
                                  rng: np.random.Generator) -> pd.DataFrame:
    """Injecte volontairement des valeurs manquantes et des outliers pour l'EDA/préprocessing."""
    df = df.copy()
    n = len(df)
    for col in cols:
        std = df[col].std()
        # outliers
        n_out = int(n * outlier_rate)
        idx_out = rng.choice(n, size=n_out, replace=False)
        signs = rng.choice([-1, 1], size=n_out)
        df.loc[df.index[idx_out], col] += signs * outlier_magnitude * std
        # valeurs manquantes
        n_miss = int(n * missing_rate)
        idx_miss = rng.choice(n, size=n_miss, replace=False)
        df.loc[df.index[idx_miss], col] = np.nan
    return df


def add_sensor_drift(series: np.ndarray, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    """Ajoute une dérive lente (tendance basse fréquence) sur 2-3 capteurs choisis."""
    n = len(series)
    n_cycles = rng.uniform(0.5, 1.5)
    phase = rng.uniform(0, 2 * np.pi)
    drift = amplitude * np.sin(np.linspace(0, 2 * np.pi * n_cycles, n) + phase)
    # dérive globalement croissante (encrassement capteur / désétalonnage) + composante cyclique
    trend = np.linspace(0, amplitude * rng.choice([-1, 1]), n)
    return series + drift * 0.4 + trend


# -----------------------------------------------------------------------------
# Génération table CDU
# -----------------------------------------------------------------------------
def generate_cdu_data(index: pd.DatetimeIndex, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    n = len(index)
    c = cfg["data_generator"]["cdu"]

    feed_rate = ou_process(n, c["feed_rate_mean"], c["feed_rate_std"] * 0.15, 0.05, rng, x0=c["feed_rate_mean"])
    ramp_mask = rng.uniform(0, 1, n) < c["feed_rate_ramp_prob"]
    ramp_targets = rng.normal(c["feed_rate_mean"], c["feed_rate_std"], n)
    for idx in np.where(ramp_mask)[0]:
        span = min(24, n - idx)
        feed_rate[idx:idx + span] = np.linspace(feed_rate[idx], ramp_targets[idx], span)

    crude_types = c["crude_types"]
    schedule = make_regime_schedule(n, c["crude_change_min_days"], c["crude_change_max_days"], rng)
    n_segments = schedule[-1] + 1
    seg_choice = rng.choice(crude_types, size=n_segments)
    crude_type = seg_choice[schedule]

    crude_api_target = np.array([c["crude_api"][ct] for ct in crude_type], dtype=np.float64)
    crude_api = smooth_transitions(crude_api_target, schedule, c["crude_transition_hours"])
    crude_api += rng.normal(0, 0.15, n)

    furnace_cot = ou_process(n, c["furnace_cot_setpoint"], c["furnace_cot_controller_noise"], 0.08, rng,
                              x0=c["furnace_cot_setpoint"])
    furnace_cot = np.clip(furnace_cot, c["furnace_cot_min"], c["furnace_cot_max"])

    column_top_temp = ou_process(n, c["column_top_temp_mean"], c["column_top_temp_std"] * 0.3, 0.1, rng,
                                  x0=c["column_top_temp_mean"])
    column_top_pressure = ou_process(n, c["column_top_pressure_mean"], c["column_top_pressure_std"] * 0.3, 0.1, rng,
                                      x0=c["column_top_pressure_mean"])
    reflux_ratio = ou_process(n, c["reflux_ratio_mean"], c["reflux_ratio_std"] * 0.3, 0.08, rng,
                               x0=c["reflux_ratio_mean"])
    stripping_steam = ou_process(n, c["stripping_steam_mean"], c["stripping_steam_std"] * 0.3, 0.1, rng,
                                  x0=c["stripping_steam_mean"])

    # --- Rendements (courbe TBP par brut + effets non linéaires COT / reflux) ---
    base = c["base_yields"]
    naphtha_base = smooth_transitions(np.array([base[ct]["naphtha"] for ct in crude_type]), schedule,
                                       c["crude_transition_hours"])
    kerosene_base = smooth_transitions(np.array([base[ct]["kerosene"] for ct in crude_type]), schedule,
                                        c["crude_transition_hours"])
    gasoil_base = smooth_transitions(np.array([base[ct]["gasoil"] for ct in crude_type]), schedule,
                                      c["crude_transition_hours"])
    residue_base = smooth_transitions(np.array([base[ct]["residue"] for ct in crude_type]), schedule,
                                       c["crude_transition_hours"])

    cot_dev = (furnace_cot - c["furnace_cot_setpoint"])
    distillate_shift = c["cot_sensitivity"] * cot_dev  # >0 si COT au dessus du setpoint -> + distillats

    naphtha = naphtha_base + distillate_shift * (naphtha_base / (naphtha_base + kerosene_base + gasoil_base))
    kerosene = kerosene_base + distillate_shift * (kerosene_base / (naphtha_base + kerosene_base + gasoil_base))
    gasoil = gasoil_base + distillate_shift * (gasoil_base / (naphtha_base + kerosene_base + gasoil_base))
    residue = residue_base - distillate_shift

    reflux_dev = (reflux_ratio - c["reflux_ratio_mean"])
    sharpness = c["reflux_sensitivity"] * reflux_dev
    naphtha = naphtha + sharpness
    kerosene = kerosene - sharpness

    yields = np.stack([naphtha, kerosene, gasoil, residue], axis=1)
    yields = np.clip(yields, 0.01, None)
    yields = yields / yields.sum(axis=1, keepdims=True)  # renormalisation stricte somme=1

    noise_pct = c["sensor_noise_pct"]
    yields = yields * (1 + rng.normal(0, noise_pct, yields.shape))
    yields = np.clip(yields, 0.001, None)
    yields = yields / yields.sum(axis=1, keepdims=True)

    df = pd.DataFrame({
        "feed_rate": feed_rate,
        "crude_type": crude_type,
        "crude_api": crude_api,
        "furnace_cot": furnace_cot,
        "column_top_temp": column_top_temp,
        "column_top_pressure": column_top_pressure,
        "reflux_ratio": reflux_ratio,
        "stripping_steam": stripping_steam,
        "naphtha_yield": yields[:, 0],
        "kerosene_yield": yields[:, 1],
        "gasoil_yield": yields[:, 2],
        "residue_yield": yields[:, 3],
    }, index=index)

    # dérive lente sur 2-3 capteurs
    for sensor in c["drift_sensors"]:
        df[sensor] = add_sensor_drift(df[sensor].values, c["drift_amplitude"], rng)

    numeric_cols = [col for col in df.columns if col != "crude_type"]
    df = inject_missing_and_outliers(df, numeric_cols, c["missing_rate"], c["outlier_rate"],
                                      c["outlier_magnitude"], rng)
    return df


# -----------------------------------------------------------------------------
# Génération table Energie / Fouling (vérité terrain cachée)
# -----------------------------------------------------------------------------
def generate_energy_data(index: pd.DatetimeIndex, cdu_df: pd.DataFrame, cfg: dict,
                          rng: np.random.Generator) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    n = len(index)
    c = cfg["data_generator"]["energy"]

    # Le nettoyage est déclenché physiquement quand la résistance d'encrassement
    # franchit `cleaning_threshold` (et non à des instants arbitraires) : la durée
    # du cycle est donc dérivée de la physique (asymptote, vitesse) avec une légère
    # variabilité opératoire d'un cycle à l'autre. Sur l'horizon de 2 ans cela produit
    # naturellement ~4-5 nettoyages comme demandé par la spécification.
    asymptote = c["fouling_asymptote"]
    threshold = c["cleaning_threshold"]
    fouling_resistance = np.zeros(n, dtype=np.float64)
    cleaning_positions: list[int] = []
    pos = 0
    pbar = tqdm(total=n, desc="Générateur | fouling (cycles encrassement/nettoyage)", unit="h")
    while pos < n:
        rate_factor = rng.uniform(0.85, 1.15)
        eff_rate = c["fouling_rate"] * rate_factor
        h_to_threshold = -np.log(1 - threshold / asymptote) / eff_rate
        seg_len = min(int(h_to_threshold), n - pos)
        t_hours = np.arange(seg_len)
        fouling_resistance[pos:pos + seg_len] = asymptote * (1 - np.exp(-eff_rate * t_hours))
        pos += seg_len
        pbar.update(seg_len)
        if pos < n:
            cleaning_positions.append(pos)
    pbar.close()
    cleaning_positions = np.array(cleaning_positions, dtype=int)
    cleaning_times = [index[p] for p in cleaning_positions]

    fouling_resistance += rng.normal(0, asymptote * 0.01, n)
    fouling_resistance = np.clip(fouling_resistance, 0, None)

    feed_rate = cdu_df["feed_rate"].fillna(cdu_df["feed_rate"].mean()).values
    feed_norm = feed_rate / cfg["data_generator"]["cdu"]["feed_rate_mean"]

    preheat_outlet_temp = (c["preheat_outlet_base"] - c["preheat_outlet_fouling_sensitivity"] * fouling_resistance
                            + rng.normal(0, c["preheat_outlet_base"] * c["sensor_noise_pct"], n))

    heat_deficit = c["preheat_outlet_base"] - preheat_outlet_temp
    fuel_gas_flow = (c["fuel_gas_flow_base"] * feed_norm + 0.02 * heat_deficit
                     + rng.normal(0, c["fuel_gas_flow_base"] * c["sensor_noise_pct"], n))
    furnace_duty = (c["furnace_duty_base_mw"] * feed_norm + 0.35 * heat_deficit
                    + rng.normal(0, c["furnace_duty_base_mw"] * c["sensor_noise_pct"], n))
    specific_energy = (c["specific_energy_base"] * (1 + 0.4 * fouling_resistance / c["fouling_asymptote"])
                        * (0.9 + 0.2 * feed_norm) + rng.normal(0, c["specific_energy_base"] * 0.01, n))
    co2_emissions = furnace_duty * c["co2_factor_t_per_mwh"]

    horizon_24 = int(24)
    horizon_48 = int(48)
    threshold = c["cleaning_threshold"]
    will_clean_24 = np.zeros(n, dtype=int)
    will_clean_48 = np.zeros(n, dtype=int)
    cleaning_bool = np.zeros(n, dtype=bool)
    cleaning_bool[np.clip(cleaning_positions, 0, n - 1)] = True
    for h, out in [(horizon_24, will_clean_24), (horizon_48, will_clean_48)]:
        for cp in cleaning_positions:
            lo = max(0, cp - h)
            out[lo:cp] = 1

    df = pd.DataFrame({
        "fouling_resistance": fouling_resistance,
        "preheat_outlet_temp": preheat_outlet_temp,
        "fuel_gas_flow": fuel_gas_flow,
        "furnace_duty": furnace_duty,
        "specific_energy": specific_energy,
        "co2_emissions": co2_emissions,
        "cleaning_needed_within_24h": will_clean_24,
        "cleaning_needed_within_48h": will_clean_48,
        "is_cleaning_event": cleaning_bool.astype(int),
    }, index=index)
    return df, cleaning_times


# -----------------------------------------------------------------------------
# Génération table Vapocraqueur
# -----------------------------------------------------------------------------
def generate_cracker_data(index: pd.DatetimeIndex, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    n = len(index)
    c = cfg["data_generator"]["cracker"]

    naphtha_feed = ou_process(n, c["naphtha_feed_mean"], c["naphtha_feed_std"] * 0.2, 0.05, rng,
                               x0=c["naphtha_feed_mean"])
    coil_outlet_temp = ou_process(n, c["coil_outlet_temp_setpoint"], 2.5, 0.08, rng,
                                   x0=c["coil_outlet_temp_setpoint"])
    coil_outlet_temp = np.clip(coil_outlet_temp, c["coil_outlet_temp_min"], c["coil_outlet_temp_max"])
    steam_to_oil_ratio = ou_process(n, c["steam_to_oil_ratio_mean"], 0.02, 0.1, rng, x0=c["steam_to_oil_ratio_mean"])
    residence_time = ou_process(n, c["residence_time_mean"], 0.01, 0.1, rng, x0=c["residence_time_mean"])

    severity = ((coil_outlet_temp - c["coil_outlet_temp_min"])
                / (c["coil_outlet_temp_max"] - c["coil_outlet_temp_min"]))
    severity = severity * (1 + 0.15 * (residence_time - c["residence_time_mean"]) / c["residence_time_mean"])
    severity = np.clip(severity, 0, 1)

    ethylene_yield = c["ethylene_yield_max"] * (1 - np.exp(-3 * severity)) + rng.normal(0, 0.004, n)
    propylene_yield = c["propylene_yield_peak"] * 4 * severity * (1 - severity) + rng.normal(0, 0.003, n)
    c4_yield = 0.12 * (1 - 0.4 * severity) + rng.normal(0, 0.003, n)
    pygas_yield = 0.18 * (1 - 0.3 * severity) + rng.normal(0, 0.004, n)

    cycle_positions = make_regime_schedule(n, c["coke_cycle_min_days"], c["coke_cycle_max_days"], rng)
    change_points = np.where(np.diff(cycle_positions) != 0)[0] + 1
    boundaries = [0] + list(change_points) + [n]
    coke_thickness = np.zeros(n, dtype=np.float64)
    for seg_start, seg_end in zip(boundaries[:-1], boundaries[1:]):
        seg_len = seg_end - seg_start
        if seg_len <= 0:
            continue
        t = np.arange(seg_len) / 24.0
        sev_seg = severity[seg_start:seg_end].mean() if seg_len > 0 else 0.5
        coke_thickness[seg_start:seg_end] = (0.3 + 0.7 * sev_seg) * (1 - np.exp(-0.05 * t))
    coke_thickness += rng.normal(0, 0.01, n)
    coke_thickness = np.clip(coke_thickness, 0, None)

    tube_metal_temp = (c["tube_metal_temp_base"] + 25 * coke_thickness + 0.4 * (coil_outlet_temp - c["coil_outlet_temp_setpoint"])
                        + rng.normal(0, c["tube_metal_temp_base"] * c["sensor_noise_pct"], n))

    df = pd.DataFrame({
        "naphtha_feed": naphtha_feed,
        "coil_outlet_temp": coil_outlet_temp,
        "steam_to_oil_ratio": steam_to_oil_ratio,
        "residence_time": residence_time,
        "ethylene_yield": ethylene_yield,
        "propylene_yield": propylene_yield,
        "c4_yield": c4_yield,
        "pygas_yield": pygas_yield,
        "coke_thickness": coke_thickness,
        "tube_metal_temp": tube_metal_temp,
    }, index=index)
    return df


# -----------------------------------------------------------------------------
# Génération table Laboratoire
# -----------------------------------------------------------------------------
def generate_lab_data(index: pd.DatetimeIndex, cdu_df: pd.DataFrame, cfg: dict,
                       rng: np.random.Generator) -> pd.DataFrame:
    c = cfg["data_generator"]["lab"]
    interval = c["sample_interval_hours"]
    delay = c["result_delay_hours"]

    sample_idx = np.arange(0, len(index), interval)
    sample_times = index[sample_idx]
    result_times = sample_times + pd.Timedelta(hours=delay)

    cot = cdu_df["furnace_cot"].values[sample_idx]
    reflux = cdu_df["reflux_ratio"].values[sample_idx]
    api = cdu_df["crude_api"].values[sample_idx]
    crude_type = cdu_df["crude_type"].values[sample_idx]

    # Coefficients calibrés pour que le signal déterministe (piloté surtout par crude_api, dont
    # l'écart-type inter-brut ~3 domine largement celui du COT ~1.6) reste bien au-dessus du
    # bruit labo, condition nécessaire pour qu'un soft sensor continu puisse approcher corr > 0.9.
    n_s = len(sample_idx)
    naphtha_fbp = 165 - 2.0 * (api - 31) + 0.4 * (cot - 365) + rng.normal(0, 1.2, n_s)
    kerosene_flash_point = 45 - 1.0 * (api - 31) + 0.2 * (cot - 365) - 0.15 * (reflux - 2.4) + rng.normal(0, 0.6, n_s)
    gasoil_cetane_index = 50 + 1.2 * (api - 31) - 0.15 * (cot - 365) + rng.normal(0, 0.6, n_s)
    residue_viscosity = 380 - 10.0 * (api - 31) + 1.5 * (cot - 365) + rng.normal(0, 6, n_s)
    sulfur_base = np.select([crude_type == "leger", crude_type == "moyen", crude_type == "lourd"],
                             [0.3, 0.9, 2.1], default=0.9)
    sulfur_content = sulfur_base * (1 + rng.normal(0, c["sulfur_noise_pct"], n_s))
    residue_viscosity = np.clip(residue_viscosity, 50, None)

    df = pd.DataFrame({
        "sample_time": sample_times,
        "result_time": result_times,
        "naphtha_final_boiling_point": naphtha_fbp,
        "kerosene_flash_point": kerosene_flash_point,
        "gasoil_cetane_index": gasoil_cetane_index,
        "residue_viscosity": residue_viscosity,
        "sulfur_content": np.clip(sulfur_content, 0.05, None),
    })
    return df


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
def generate_all(cfg: dict | None = None) -> dict[str, pd.DataFrame]:
    cfg = cfg or load_config()
    set_global_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    dg = cfg["data_generator"]
    index = pd.date_range(start=dg["start_date"], end=dg["end_date"], freq=dg["freq"])

    steps = tqdm(total=4, desc="Générateur de données synthétiques", unit="table")

    cdu_df = generate_cdu_data(index, cfg, rng)
    steps.update(1)

    energy_df, cleaning_times = generate_energy_data(index, cdu_df, cfg, rng)
    steps.update(1)

    cracker_df = generate_cracker_data(index, cfg, rng)
    steps.update(1)

    lab_df = generate_lab_data(index, cdu_df, cfg, rng)
    steps.update(1)
    steps.close()

    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    cdu_df.to_csv(raw_dir / "cdu_data.csv", index_label="timestamp")
    energy_df.to_csv(raw_dir / "energy_data.csv", index_label="timestamp")
    cracker_df.to_csv(raw_dir / "cracker_data.csv", index_label="timestamp")
    lab_df.to_csv(raw_dir / "lab_data.csv", index=False)

    print(f"[data_generator] {len(cdu_df)} lignes générées pour cdu_data / energy_data / cracker_data")
    print(f"[data_generator] {len(lab_df)} échantillons labo générés")
    print(f"[data_generator] Nettoyages (vérité terrain, jamais en feature) : "
          f"{[t.strftime('%Y-%m-%d %H:%M') for t in cleaning_times]}")
    print(f"[data_generator] Fichiers écrits dans {raw_dir}")

    return {"cdu": cdu_df, "energy": energy_df, "cracker": cracker_df, "lab": lab_df}


if __name__ == "__main__":
    generate_all()
