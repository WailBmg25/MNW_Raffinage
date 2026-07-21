"""
train_all.py
=============
CLI bonus : régénère les données, reconstruit les artefacts de préprocessing, puis exécute
l'intégralité des notebooks d'entraînement (03 à 06) — chacun affiche déjà sa propre
progression détaillée par epoch via tqdm (src/training.py). Les artefacts (.pt, scalers,
JSON de synthèse) sont écrits directement dans backend/models_artifacts/ par les notebooks ;
ce script vérifie leur présence à la fin et affiche un récapitulatif.

Usage :
    python train_all.py                  # génère les données si absentes, puis entraîne tout
    python train_all.py --regenerate-data  # force la régénération des données synthétiques
    python train_all.py --skip-notebooks 01,02  # notebooks à ignorer (par défaut 01,02 ignorés,
                                                  # ce sont des notebooks d'exploration, pas d'entraînement)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
ARTIFACTS_DIR = PROJECT_ROOT / "backend" / "models_artifacts"

TRAINING_NOTEBOOKS = [
    "03_yield_prediction.ipynb",
    "04_fouling_detection.ipynb",
    "05_energy_optimization.ipynb",
    "06_realtime_system.ipynb",
]

EXPECTED_ARTIFACTS = [
    "yields_best.pt", "yields_best_model.json", "yields_scaler_X.joblib",
    "fouling_production_model.json", "fouling_scaler_X.joblib",
    "surrogate_energy.pt", "surrogate_scaler_X.joblib", "energy_optimization_summary.json",
    "quality_soft_sensor.pt", "quality_scaler_X.joblib", "quality_scaler_y.joblib",
]


def run(cmd: list[str], step_name: str) -> None:
    print(f"\n{'=' * 70}\n>>> {step_name}\n{'=' * 70}")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"\n[train_all] ÉCHEC à l'étape : {step_name}")
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraîne tous les modèles DL du projet.")
    parser.add_argument("--regenerate-data", action="store_true",
                        help="Force la régénération des données synthétiques même si data/raw existe déjà.")
    parser.add_argument("--skip-notebooks", type=str, default="",
                        help="Liste de suffixes de notebooks à ignorer, séparés par des virgules (ex: 04,05).")
    args = parser.parse_args()

    skip = {s.strip() for s in args.skip_notebooks.split(",") if s.strip()}
    notebooks_to_run = [nb for nb in TRAINING_NOTEBOOKS if not any(nb.startswith(s) for s in skip)]

    raw_dir = PROJECT_ROOT / "data" / "raw"
    needs_data = args.regenerate_data or not (raw_dir / "cdu_data.csv").exists()

    steps = []
    if needs_data:
        steps.append(("Génération des données synthétiques", [sys.executable, "-m", "src.data_generator"]))
    steps.append(("Préprocessing (nettoyage, séquences, split, scalers)",
                  [sys.executable, "-m", "src.preprocessing"]))
    for nb in notebooks_to_run:
        steps.append((f"Entraînement — notebook {nb}", [
            sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
            str(NOTEBOOKS_DIR / nb), "--ExecutePreprocessor.timeout=3600",
        ]))

    for step_name, cmd in tqdm(steps, desc="Pipeline d'entraînement complet", unit="étape"):
        run(cmd, step_name)

    print(f"\n{'=' * 70}\n>>> Vérification des artefacts dans {ARTIFACTS_DIR}\n{'=' * 70}")
    missing = [f for f in EXPECTED_ARTIFACTS if not (ARTIFACTS_DIR / f).exists()]
    for f in EXPECTED_ARTIFACTS:
        status = "✅" if (ARTIFACTS_DIR / f).exists() else "❌"
        print(f"  {status} {f}")

    if missing:
        print(f"\n[train_all] ATTENTION : {len(missing)} artefact(s) manquant(s) — "
              f"vérifier les notebooks correspondants ({', '.join(missing)}).")
        sys.exit(1)

    report_path = PROJECT_ROOT / "data" / "results" / "model_report.md"
    print(f"\n[train_all] Tous les artefacts sont présents. Rapport final : {report_path}")
    if report_path.exists():
        print(report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
