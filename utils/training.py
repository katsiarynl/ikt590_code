import os
import numpy as np
from time import time


def train_tm_and_save_scores(
    TMClassifier,
    config,
    X_train,
    Y_train,
    X_test,
    Y_test,
    T,
    s,
    patch_dim,
    max_included_literals,
    weighted_clauses,
    specialist_name,
    save_dir,
    include_patch_size=True,
):
    """
    Trains a Tsetlin Machine for N epochs and saves class sums per epoch.

    Returns:
        tm (trained model)
    """
    if include_patch_size:
        tm = TMClassifier(
            number_of_clauses=config.NUM_CLAUSES*2,
            T=T,
            s=s,
            max_included_literals=max_included_literals,
            platform="CPU",
            weighted_clauses=weighted_clauses,
            patch_dim=(patch_dim, patch_dim),
        )
    else:
        tm = TMClassifier(
            number_of_clauses=config.NUM_CLAUSES*2,
            T=T,
            s=s,
            max_included_literals=max_included_literals,
            platform="CPU",
            weighted_clauses=weighted_clauses,
        )

    os.makedirs(save_dir, exist_ok=True)

    print(f"\nAccuracy over {config.N_EPOCHS} epochs:\n")

    for epoch in range(config.N_EPOCHS):
        start = time()

        tm.fit(X_train, Y_train, epochs=1, incremental=True)

        stop = time()

        Y_pred, class_sums = tm.predict(X_test, return_class_sums=True)

        acc = 100.0 * (Y_pred == Y_test).mean()
        print(f"#{epoch+1} Accuracy: {acc:.2f}% ({stop-start:.2f}s)")

        file_path = os.path.join(
            save_dir,
            f"{specialist_name}_{epoch+1}_"
            f"{config.NUM_CLAUSES}_{T}_{s:.1f}_"
            f"{patch_dim}_{max_included_literals}_{weighted_clauses}.txt"
        )

        np.savetxt(file_path, class_sums, delimiter=",")

    return tm
