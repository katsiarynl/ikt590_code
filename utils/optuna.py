import json
import os
from datetime import datetime
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import f1_score, fbeta_score, accuracy_score


def save_optuna_best_results(
    study,
    save_path="results/best_optuna_results.json",
    append=False,
):
    """
    Saves the best Optuna study results (best value + best params) to JSON.

    Args:
        study: Optuna study object
        save_path (str): Where to save the file
        append (bool): If True, appends to existing file (as a list)
                       If False, overwrites file
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    result_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_value": study.best_value,
        "best_params": study.best_params,
    }

    # If appending and file exists → load old results
    if append and os.path.exists(save_path):
        with open(save_path, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = [data]
        data.append(result_entry)
    else:
        data = result_entry if not append else [result_entry]

    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Best Optuna results saved to {save_path}")

import json


def load_optuna_best_results(load_path):
    """
    Loads best Optuna results from JSON file.

    Returns:
        best_value (float)
        best_params (dict)
    """

    with open(load_path, "r") as f:
        data = json.load(f)

    # If file contains multiple runs (append=True case)
    if isinstance(data, list):
        data = data[-1]  # get most recent entry

    best_value = data["best_value"]
    best_params = data["best_params"]

    return best_value, best_params





def run_tm_optuna_search(
    TMClassifier,
    config,
    X_train,
    Y_train,
    X_val,
    Y_val,
    *,
    metric="f1_macro",
    pos_label=1,
    include_patch_size=True,
):
    """
    Generic Optuna search using values from config.py

    metric:
        "f1_macro"
        "accuracy"
        "fbeta"
    """

    def objective(trial):


        s_cand = trial.suggest_float("s", 1.0, 15.0, step=1.0)
        T_cand = trial.suggest_int("T", 1000, 10000, step=500)
        if include_patch_size:
            patch_cand = trial.suggest_categorical("patch_size", [3, 5, 7, 9])
        w_cand = 1
        maxlit_c = trial.suggest_categorical("max_included_literals",[32, 64, 128, 512])

        
        if include_patch_size:
            tm_trial = TMClassifier(
                number_of_clauses=config.NUM_CLAUSES,
                T=T_cand,
                s=s_cand,
                max_included_literals=maxlit_c,
                platform=config.DEVICE,
                weighted_clauses=w_cand,
                patch_dim=(patch_cand, patch_cand),
            )
        else:
            tm_trial = TMClassifier(
                number_of_clauses=config.NUM_CLAUSES,
                T=T_cand,
                s=s_cand,
                max_included_literals=maxlit_c,
                platform=config.DEVICE,
                weighted_clauses=w_cand,
            )

        # --- Training ---
        for _ in range(config.N_EPOCHS_PER_TRIAL):
            tm_trial.fit(X_train, Y_train, epochs=1, incremental=True)

        # --- Validation ---
        _, Y_val_scores = tm_trial.predict(
            X_val,
            return_class_sums=True,
        )

        Y_pred = Y_val_scores.argmax(axis=1)

        if metric == "accuracy":
            score = accuracy_score(Y_val, Y_pred)
        # --- Metric selection ---
        elif metric == "f1_macro":
            score = f1_score(Y_val, Y_pred, average="macro")
        elif metric == "fbeta":
            score = fbeta_score(
                Y_val,
                Y_pred,
                beta=config.BETA,
                pos_label=pos_label,
            )

        else:
            raise ValueError("Unsupported metric type.")

        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=config.RANDOM_SEED),
    )

    study.optimize(objective, n_trials=config.N_TRIALS)

    print(f"Best {metric}: {study.best_value}")
    print("Best hyperparameters:", study.best_params)

    return study
