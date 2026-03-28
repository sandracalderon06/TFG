import os
import random
import json
import hashlib
import shutil
import joblib

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report, 
    ConfusionMatrixDisplay, roc_auc_score
)
import torch

def train_knn_experiment(dataset, experiment_name, n_neighbors=1, metric="euclidean", seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    data_path = os.path.join("./experiments/data", dataset)
    X_train = np.load(os.path.join(data_path, "X_train.npy"))
    y_train = np.load(os.path.join(data_path, "y_train.npy"))
    X_test = np.load(os.path.join(data_path, "X_test.npy"))
    y_test = np.load(os.path.join(data_path, "y_test.npy"))

    

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # CREO EL HASH DEL EXPERIMENTO
    exp_hash = hashlib.sha1(f"{n_neighbors}_{metric}_{seed}".encode()).hexdigest()
    exp_hash_folder = os.path.join("./experiments/models", dataset, experiment_name, exp_hash)
    os.makedirs(exp_hash_folder, exist_ok=True)


    model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    model.fit(X_train_flat, y_train)

    joblib.dump(model, os.path.join(exp_hash_folder, "model.pkl"))

    train_params = {
        "experiment_hash": exp_hash,
        "n_neighbors": n_neighbors,
        "metric": metric,
        "seed": seed,
        "total_params": 0  # k-NN no tiene parámetros entrenables
    }
    with open(os.path.join(exp_hash_folder, "train_params.json"), "w") as f:
        json.dump(train_params, f)

    y_train_pred = model.predict(X_train_flat)
    y_test_pred = model.predict(X_test_flat)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred, average="weighted")
    test_f1 = f1_score(y_test, y_test_pred, average="weighted")

    if len(np.unique(y_train)) == 2:
        y_train_prob = model.predict_proba(X_train_flat)[:,1]
        y_test_prob = model.predict_proba(X_test_flat)[:,1]
        train_roc_auc = roc_auc_score(y_train, y_train_prob)
        test_roc_auc = roc_auc_score(y_test, y_test_prob)
    else:
        y_train_prob = model.predict_proba(X_train_flat)
        y_test_prob = model.predict_proba(X_test_flat)
        train_roc_auc = roc_auc_score(y_train, y_train_prob, multi_class="ovr", average="macro")
        test_roc_auc = roc_auc_score(y_test, y_test_prob, multi_class="ovr", average="macro")

    metrics = {
        "train_acc": train_acc, "test_acc": test_acc,
        "train_f1": train_f1, "test_f1": test_f1,
        "train_roc_auc": train_roc_auc, "test_roc_auc": test_roc_auc
    }
    with open(os.path.join(exp_hash_folder, "metrics.json"), "w") as f:
        json.dump(metrics, f)

    # Guardar resultados
    results_df = pd.DataFrame({"y_test": y_test, "y_pred": y_test_pred})
    results_df.to_excel(os.path.join(exp_hash_folder, "results.xlsx"), index=False)

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_test_pred)
    cmp = ConfusionMatrixDisplay(cm)
    fig, ax = plt.subplots(figsize=(8,8))
    cmp.plot(ax=ax)
    plt.title(f"Confusion Matrix: {experiment_name}")
    plt.savefig(os.path.join(exp_hash_folder, "confusion_matrix.png"))
    plt.close(fig)

    # Loss curve
    fig, ax = plt.subplots()
    ax.plot([0,1], [1-train_acc, 1-test_acc], label="pseudo_loss")
    ax.set_title("Loss Curve (pseudo)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    plt.savefig(os.path.join(exp_hash_folder, "loss_curve.png"))
    plt.close(fig)

    main_folder = os.path.join("./experiments/models", dataset, experiment_name)
    shutil.copyfile(os.path.join(exp_hash_folder, "model.pkl"), os.path.join(main_folder, "model.pkl"))
    shutil.copyfile(os.path.join(exp_hash_folder, "train_params.json"), os.path.join(main_folder, "train_params.json"))

    all_results = pd.DataFrame([train_params | metrics])
    all_results.to_excel(os.path.join(main_folder, "all_results.xlsx"), index=False)
    all_results.to_excel(os.path.join(main_folder, "average_seed_results.xlsx"), index=False)

    accuracy = accuracy_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred, average="weighted")
    print(f"Experiment {experiment_name} | k={n_neighbors}, metric={metric}")
    print(f"Accuracy: {accuracy*100:.2f}% | F1-score: {f1*100:.2f}%")
    print(classification_report(y_test, y_test_pred))

    print("Finished")

    return exp_hash_folder, main_folder