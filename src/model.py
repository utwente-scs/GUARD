
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Optional

from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Frozen method constants. These values define the method under evaluation
# and are not tunable; any change invalidates the reported results.
# ---------------------------------------------------------------------------
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DECISION_THRESHOLD = 0.40
FIEX_C = 0.01

LEAF_COL = "leaf"
LABEL_COL = "label"
NUM_COLS = ["label", "n_true", "n_false", "n_unl", "n_labeled", "pos_frac"]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dataset(path: str) -> pd.DataFrame:
    
    df = pd.read_csv(path, keep_default_na=False, na_values=[])
    df[LEAF_COL] = df[LEAF_COL].astype(str)
    df = df[df[LEAF_COL].str.strip() != ""].reset_index(drop=True)
    for col in NUM_COLS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df[LABEL_COL] = df[LABEL_COL].fillna(-1).astype(int)
    return df


def load_embedding_cache(path: str, df: pd.DataFrame) -> np.ndarray:

    X = np.load(path)
    if X.shape[0] != len(df):
        raise ValueError(
            f"Embedding cache misaligned: {X.shape[0]} rows vs {len(df)} "
            f"dataset rows. The CSV must be loaded via load_dataset().")
    return X


def get_embedder(opt_model: Optional[str] = None):
    
    if opt_model is None:
        model = SentenceTransformer(MODEL_NAME)
    else:
        model = SentenceTransformer(opt_model)

    def embed(texts):
        return np.asarray(
            model.encode(list(texts), batch_size=256, show_progress_bar=True))
    return embed

class GUARD:
    
    def __init__(self, C=FIEX_C, class_weight=None, solver="lbfgs",max_iter=5000):

        self.C = C
        self.class_weight = class_weight
        self.solver = solver
        self.max_iter = max_iter

    def fit(self, X_all_raw: np.ndarray, y_all: np.ndarray, train_idx=None,):
        self.scaler = StandardScaler().fit(X_all_raw)  # full canonical corpus
        self.X_all = self.scaler.transform(X_all_raw)
        self.y_all = np.asarray(y_all).astype(int)
        if train_idx is None:
            self.labeled_idx = np.where(self.y_all != -1)[0]

        else:
            self.labeled_idx = np.asarray(train_idx)
        self.clf = LogisticRegression(C=self.C, class_weight=self.class_weight,solver=self.solver,max_iter=self.max_iter)
        self.clf.fit(self.X_all[self.labeled_idx], self.y_all[self.labeled_idx])
        return self

    def transform_raw(self, X_raw: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X_raw)

    def predict_proba_raw(self, X_raw: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(self.transform_raw(X_raw))[:, 1]

    def predict_raw(self, X_raw: np.ndarray) -> np.ndarray:
        return (self.predict_proba_raw(X_raw)
                >= DECISION_THRESHOLD).astype(int)

    def sanity_in_sample(self):
        
        Xl = self.X_all[self.labeled_idx]
        yl = self.y_all[self.labeled_idx]
        p = self.clf.predict_proba(Xl)[:, 1]
        rec = float(((p >= DECISION_THRESHOLD)[yl == 1]).mean())
        return rec, float(p.max())