
import os
import numpy as np
import pandas as pd

FOLDER = "../../resources/datasets/"

DATASETS = {
    "t3": f"{FOLDER}dataset_t3.csv",
}

LEAF_COL = "leaf"
MODEL_NAME = "all-mpnet-base-v2"
FORCE_REBUILD = False
NUM_COLS = ["label", "n_true", "n_false", "n_unl", "n_labeled", "pos_frac"]


def load_variant(path: str) -> pd.DataFrame:
    # keep literal leaf names like "null"/"nan"; coerce numeric columns explicitly
    df = pd.read_csv(path, keep_default_na=False, na_values=[])
    df[LEAF_COL] = df[LEAF_COL].astype(str)
    df = df[df[LEAF_COL].str.strip() != ""].reset_index(drop=True)

    for col in NUM_COLS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["label"] = df["label"].fillna(-1).astype(int)
    if "hc_neg" in df:
        df["hc_neg"] = df["hc_neg"].astype(str).str.lower().isin(["true", "1"])
    else:
        df["hc_neg"] = False
    return df


_shared_embeddings = None  # leaves are identical across variants: encode once

for tag, path in DATASETS.items():
    if not os.path.exists(path):
        print(f"[{tag}] missing: {path} -- skipped\n")
        continue

    df = load_variant(path)
    assert df[LEAF_COL].is_unique, f"[{tag}] expected one row per unique leaf"

    y = df["label"].to_numpy()
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    n_hc = int(df["hc_neg"].sum())
    n_unl = int((y == -1).sum())
    n_lab = n_pos + n_neg

    print(f"===== [{tag}] {path} =====")
    print(f"Unique leaves: {len(df)}")
    print(f"Labeled: {n_lab} | pos: {n_pos} ({n_pos / n_lab:.1%}) "
          f"| neg: {n_neg} (annotated: {n_neg - n_hc}, heuristic: {n_hc}) "
          f"| unlabeled: {n_unl}")
    print(f"Imbalance ratio (neg:pos): {n_neg / max(n_pos, 1):.2f}:1")

