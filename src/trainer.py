
import argparse
import os
import pickle

import numpy as np

from model import (
    GUARD,
    LABEL_COL,
    LEAF_COL,
    MODEL_NAME,
    load_dataset,
    load_embedding_cache,
    get_embedder,
)

MODEL_FILENAME = "guard_model.pkl"


def build_embeddings(df, cache_path=None, save_path=None):
    
    if cache_path is not None:
        print(f"[trainer] loading cached embeddings from {cache_path}")
        return load_embedding_cache(cache_path, df)

    print(f"[trainer] embedding {len(df)} key names with {MODEL_NAME} "
          "(this downloads the model on first run)")
    embed = get_embedder()
    X = embed(df[LEAF_COL].tolist())

    if save_path is not None:
        np.save(save_path, X)
        print(f"[trainer] embedding cache saved to {save_path}")
    return X


def main():
    parser = argparse.ArgumentParser(
        description="Train the GUARD hardcoded-secret model and save the "
                    "pretrained artifact.")
    parser.add_argument("-d", "--dataset", required=True,
                        help="canonical dataset CSV (columns: leaf, label; "
                             "label in {1, 0, -1} where -1 = unlabeled).")
    parser.add_argument("-e", "--embeddings", default=None,
                        help="optional cached .npy embedding matrix, "
                             "row-aligned with the dataset CSV.")
    parser.add_argument("--save-embeddings", default=None,
                        help="optional path to save the freshly computed "
                             "embedding matrix as .npy for later reuse.")
    parser.add_argument("-o", "--output-dir", default="./pretrainedModels",
                        help="folder where the pretrained model is saved "
                             "(default: ./pretrainedModels).")
    args = parser.parse_args()

    #Data ---------------------------------------------------------------
    df = load_dataset(args.dataset)
    y_all = df[LABEL_COL].to_numpy()
    n_labeled = int((y_all != -1).sum())
    print(f"[trainer] dataset: {len(df)} rows "
          f"({n_labeled} labeled, {len(df) - n_labeled} unlabeled)")

    X_all = build_embeddings(df, cache_path=args.embeddings,
                             save_path=args.save_embeddings)

    pipe = GUARD().fit(X_all, y_all)

    rec, pmax = pipe.sanity_in_sample()
    print(f"[trainer] in-sample sanity: recall={rec:.3f}, max_proba={pmax:.3f}")
    if rec < 0.70 or pmax < 0.90:
        print("[trainer] WARNING: sanity values are well below the reference "
              "(~0.79 / ~0.99). Feature geometry may be broken (encoder "
              "mismatch or missing standardization); do not ship this model.")

    # Save ---------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, MODEL_FILENAME)
    with open(out_path, "wb") as f:
        pickle.dump(pipe, f)
    print(f"[trainer] pretrained model saved to {out_path}")


if __name__ == "__main__":
    main()