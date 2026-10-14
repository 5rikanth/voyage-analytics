
"""Attach existing Voyage Analytics model files to MLflow runs.

Uses the running MLflow server so artifacts are written to the
Docker-mounted artifact directory. Does not retrain models.
"""

from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

TRACKING_URI = "http://localhost:5001"
EXPERIMENT_NAME = "voyage-analytics"

# Existing run names and their corresponding saved files.
ARTIFACTS = {
    "flight_price_regression": [
        "flight_price_model.joblib",
        "metrics_flight_price.json",
        "route_table.json",
    ],
    "gender_classification": [
        "gender_model.joblib",
        "metrics_gender.json",
    ],
    "hotel_recommender": [
        "hotel_recommender.joblib",
        "metrics_recommender.json",
    ],
}

def main():
    client = MlflowClient(tracking_uri=TRACKING_URI)

    experiments = client.search_experiments()
    matches = [
        exp for exp in experiments
        if exp.name == EXPERIMENT_NAME
    ]

    if not matches:
        raise RuntimeError(
            f"Experiment not found: {EXPERIMENT_NAME}"
        )

    experiment = matches[0]
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=1000,
    )

    # Index existing runs by their MLflow run name.
    by_name = {}
    for run in runs:
        name = run.data.tags.get("mlflow.runName", "")
        by_name.setdefault(name, []).append(run)

    for run_name, filenames in ARTIFACTS.items():
        candidates = by_name.get(run_name, [])

        if not candidates:
            print(f"\nSKIP: No run named {run_name!r}")
            continue

        # Use the latest matching parent run.
        run = max(
            candidates,
            key=lambda item: item.info.start_time or 0,
        )
        run_id = run.info.run_id

        print(f"\nRun: {run_name} ({run_id})")

        for filename in filenames:
            path = MODELS / filename

            if not path.is_file():
                print(f"  MISSING: {path}")
                continue

            client.log_artifact(
                run_id=run_id,
                local_path=str(path),
                artifact_path="outputs",
            )
            print(f"  Uploaded: {filename}")

    print("\nFinished. Refresh the MLflow run pages.")

if __name__ == "__main__":
    main()
