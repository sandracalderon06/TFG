import os
from experiments.models.knn_utils import train_knn_experiment
from experiments.experiment_utils import load_parameters_from_json, generate_settings_combinations

DATASETS = ["ecg"]
PARAMS_PATH = "experiments/params_model_training/experiment_knn.json"

if __name__ == "__main__":

    all_params = load_parameters_from_json(PARAMS_PATH)
    experiment_name = all_params['experiment_name']
    params_combinations = generate_settings_combinations(all_params)

    for dataset in DATASETS:
        for experiment_hash, experiment_params in params_combinations.items():
            exp_name = f"{experiment_name}"

            print(f"Starting experiment {exp_name} for dataset {dataset}...")
            try:
                train_knn_experiment(
                    dataset=dataset,
                    experiment_name=exp_name,
                    n_neighbors=experiment_params["n_neighbors"],
                    metric=experiment_params["metric"]
                )
            except (ValueError, FileNotFoundError, TypeError) as msg:
                print(msg)

    print("Finished KNN experiments")