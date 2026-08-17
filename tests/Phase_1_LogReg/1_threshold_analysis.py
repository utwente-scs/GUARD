
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import GUARD, LEAF_COL, get_embedder, load_dataset  # noqa: E402
from splits import get_groups, inner_splits, make_partition, mask_labels  # noqa: E402

# ---------------------------------------------------------
# CONFIG 
# ---------------------------------------------------------

CSV_PATH = "../../resources/datasets/dataset_t3.csv"
OUTPUT_PATH = "../../resources/out/threshold_results.csv"

MODEL = "sentence-transformers/all-mpnet-base-v2"

N_SWEEPS = 30
VAL_FRACTION = 0.2
SEED_OFFSET = 200

THRESHOLDS = np.arange(0.05, 1.00, 0.05)


def compute_metrics(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "f1": f1_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
    }


def main():
    df = load_dataset(CSV_PATH)
    assert df[LEAF_COL].is_unique, "duplicate keys in dataset"
    leaves = df[LEAF_COL].to_numpy()
    y = df["label"].to_numpy()

    select_idx, holdout_idx = make_partition(df, y)
    groups = get_groups(df)
    y_sel = mask_labels(y, holdout_idx)

    embed = get_embedder(MODEL)
    X_raw = embed(leaves)

    all_proba, all_labels = [], []

    for fit_idx, val_idx in inner_splits(
        select_idx, y, groups, N_SWEEPS, VAL_FRACTION, SEED_OFFSET
    ):
        guard = GUARD()
        guard.fit(X_raw, y_sel, train_idx=fit_idx)

        all_proba.extend(guard.predict_proba_raw(X_raw[val_idx]))
        all_labels.extend(y[val_idx])

    all_proba = np.asarray(all_proba)
    all_labels = np.asarray(all_labels)
    print(f"Validation predictions pooled over {N_SWEEPS} sub-splits: "
          f"{len(all_labels)} (keys recur across sub-splits, so these are "
          f"not independent samples)")

    results = pd.DataFrame(
        [compute_metrics(all_labels, all_proba, t) for t in THRESHOLDS]
    )
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print("\nThreshold sweep:")
    print(results.to_string(index=False))

    best = results.loc[results["f1"].idxmax()]
    print("\nSelected threshold (argmax F1 on SELECT):")
    print(best.to_string())

    fixed = results.loc[np.isclose(results["threshold"], 0.5)]
    print("\nReference, threshold 0.5:")
    print(fixed.to_string(index=False))

    print("\nCopy the selected threshold into script 2.")


if __name__ == "__main__":
    main()