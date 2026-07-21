"""Optimisation énergétique par descente de gradient sur les entrées (COT, reflux), à
poids du surrogate gelés — module réutilisable par le backend (endpoint on-demand) et le
notebook 06. Le notebook 05 explore la méthode en détail à des fins pédagogiques ; ce
module fournit la version "production" appelée à la demande."""
from __future__ import annotations

import numpy as np
import torch

from src.config_loader import load_config
from src.model_registry import ModelRegistry

YIELD_IDX = slice(0, 4)   # naphta, kerosene, gasoil, residue
ENERGY_IDX = 4            # specific_energy


class EnergyOptimizer:
    def __init__(self, registry: ModelRegistry, cfg: dict | None = None):
        self.registry = registry
        self.cfg = cfg or load_config()
        self.eo_cfg = self.cfg["energy_optimization"]

    @property
    def available(self) -> bool:
        return self.registry.surrogate.available

    def predict(self, X_row: np.ndarray) -> np.ndarray | None:
        return self.registry.predict_surrogate(X_row)

    def build_row(self, base_row, conditions: dict) -> np.ndarray:
        """Construit un vecteur de features (1, n_features) dans l'ordre attendu par le
        surrogate, à partir d'une ligne de référence (dernières conditions réelles connues)
        et des conditions "what-if" fournies par l'utilisateur (feed_rate, crude_type,
        furnace_cot, reflux_ratio, ...)."""
        feature_names = self.registry.surrogate.feature_names
        crude_api_map = self.cfg["data_generator"]["cdu"]["crude_api"]
        crude_type = conditions.get("crude_type")

        row = {}
        for name in feature_names:
            if name.startswith("crude_"):
                ctype = name.replace("crude_", "")
                row[name] = 1.0 if crude_type == ctype else 0.0
            elif name == "crude_api" and crude_type in crude_api_map:
                row[name] = crude_api_map[crude_type]
            elif name in conditions and conditions[name] is not None:
                row[name] = conditions[name]
            else:
                row[name] = float(base_row[name]) if name in base_row else 0.0
        return np.array([[row[name] for name in feature_names]], dtype=np.float32)

    def optimize(self, X_row: np.ndarray, cot_idx: int, reflux_idx: int) -> dict:
        """X_row : (1, n_features) conditions actuelles NON scalées. `cot_idx`/`reflux_idx`
        sont les positions de furnace_cot / reflux_ratio dans le vecteur de features."""
        bundle = self.registry.surrogate
        if not bundle.available:
            raise RuntimeError("Surrogate non disponible (mode dégradé)")

        device = self.registry.device
        scaler = bundle.scaler
        mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
        scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)

        x_raw = torch.tensor(X_row, dtype=torch.float32, device=device).clone()
        cot0 = float(x_raw[0, cot_idx].item())
        reflux0 = float(x_raw[0, reflux_idx].item())

        for p in bundle.model.parameters():
            p.requires_grad_(False)

        cot_param = torch.nn.Parameter(torch.tensor([cot0], device=device))
        reflux_param = torch.nn.Parameter(torch.tensor([reflux0], device=device))
        optimizer = torch.optim.Adam([cot_param, reflux_param], lr=self.eo_cfg["gradient_lr"])

        cot_lo, cot_hi = self.eo_cfg["cot_bounds"]
        reflux_lo, reflux_hi = self.eo_cfg["reflux_bounds"]

        with torch.no_grad():
            baseline_out = bundle.model((x_raw - mean) / scale)
            baseline_distillate = baseline_out[0, YIELD_IDX].sum().item()
            baseline_energy = baseline_out[0, ENERGY_IDX].item()

        relu = torch.nn.functional.relu
        for _ in range(self.eo_cfg["gradient_steps"]):
            optimizer.zero_grad()
            x = x_raw.clone()
            x[0, cot_idx] = cot_param[0]
            x[0, reflux_idx] = reflux_param[0]
            x_scaled = (x - mean) / scale
            out = bundle.model(x_scaled)
            distillate = out[0, YIELD_IDX].sum()
            energy = out[0, ENERGY_IDX]

            yield_penalty = relu(baseline_distillate - self.eo_cfg["min_distillate_drop_pt"] / 100 - distillate)
            bounds_penalty = (relu(cot_lo - cot_param[0]) + relu(cot_param[0] - cot_hi)
                               + relu(reflux_lo - reflux_param[0]) + relu(reflux_param[0] - reflux_hi))
            loss = energy + self.eo_cfg["yield_penalty_weight"] * yield_penalty \
                + self.eo_cfg["bounds_penalty_weight"] * bounds_penalty
            loss.backward()
            optimizer.step()

        cot_rec = float(np.clip(cot_param.item(), cot_lo, cot_hi))
        reflux_rec = float(np.clip(reflux_param.item(), reflux_lo, reflux_hi))

        with torch.no_grad():
            x_final = x_raw.clone()
            x_final[0, cot_idx] = cot_rec
            x_final[0, reflux_idx] = reflux_rec
            final_out = bundle.model((x_final - mean) / scale)
            final_distillate = final_out[0, YIELD_IDX].sum().item()
            final_energy = final_out[0, ENERGY_IDX].item()

        gain_pct = 100.0 * (baseline_energy - final_energy) / max(baseline_energy, 1e-9)
        capacity_bpd = self.cfg["data_generator"]["refinery_capacity_bpd"]
        kwh_saved_per_bbl = max(baseline_energy - final_energy, 0.0)
        mwh_per_day = kwh_saved_per_bbl * capacity_bpd / 1000.0
        eur_per_day = mwh_per_day * self.eo_cfg["gas_price_eur_per_mwh"]
        tco2_per_day = mwh_per_day * self.eo_cfg["co2_factor_t_per_mwh"]
        constraints_ok = final_distillate >= baseline_distillate - self.eo_cfg["min_distillate_drop_pt"] / 100 - 1e-3

        return {
            "cot_current": cot0, "cot_recommended": cot_rec,
            "reflux_current": reflux0, "reflux_recommended": reflux_rec,
            "gain_pct": float(gain_pct), "eur_per_day": float(eur_per_day),
            "tco2_per_day": float(tco2_per_day), "constraints_ok": bool(constraints_ok),
        }
