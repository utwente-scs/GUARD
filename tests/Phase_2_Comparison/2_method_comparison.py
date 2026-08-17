
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import model  

warnings.filterwarnings("ignore")

# ---------------- Config ----------------
CSV_PATH = "../../resources/datasets/dataset_t3.csv"
EMB_CACHE = "../../resources/datasets/embeddings_cache.npy"
EMB_MODEL = None
LEAF_COL = model.LEAF_COL
GROUP_COL = "project"

N_SWEEPS = 30
EVAL_FRACTION = 0.30
SEED_OFFSET = 100 
MAX_ITER = 500

SUMMARY_PATH = "../../resources/out/comparison_summary.csv"
PER_RUN_PATH = "../../resources/out/comparison_per_run.csv"

PRED_DIR = "./predictions"
LLM_CACHES = {
    "LLM claude-sonnet-4-6 (zero-shot)":
        "llm_predictions_secrets_t3_claude-sonnet-4-6.csv",
    "LLM claude-fable-5 (zero-shot)":
        "llm_predictions_secrets_t3_claude-fable-5.csv",
    "LLM mistral:7b-instruct q4 local (zero-shot)":
        "llm_predictions_secrets_t3_mistral.csv",
    "GLITCH RegEx":
        "GLITCH_predictions_secrets_t3.csv",
    "IntellSia":
        "intellsia_predictions_secrets_t3.csv",
}


FIXED_KNN = (21, "distance")
FIXED_RF = (20, 300, "balanced")
FIXED_LP_K = 13
FIXED_LS = (13, 0.5)

# ---------------- Helpers ----------------

def pos_metrics(y_true, y_pred):
    return (f1_score(y_true, y_pred, pos_label=1, zero_division=0),
            precision_score(y_true, y_pred, pos_label=1, zero_division=0),
            recall_score(y_true, y_pred, pos_label=1, zero_division=0))


def counts(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp, fp, fn


def load_pred_cache(path):
    if not os.path.exists(path):
        return None
    cache = pd.read_csv(path, dtype={"leaf": str},
                        keep_default_na=False, na_values=[])
    return dict(zip(cache["leaf"], cache["pred"].astype(int)))


def get_groups(df, group_col, leaf_col):
    
    if group_col in df.columns:
        return df[group_col].to_numpy(), f"column '{group_col}'"
    else:
        
        groups = df[leaf_col].astype(str).str.lower().str.replace(r'[^a-z0-9]', '', regex=True).to_numpy()
        return groups, "normalized leaf"


def mask_labels(y, eval_idx):
    
    y_masked = y.copy()
    y_masked[eval_idx] = -1
    return y_masked


def generate_splits(labeled_idx, y, groups, n_splits, test_size, random_state):
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    labeled_groups = groups[labeled_idx]
    
    splits = []
    
    for tr_rel, te_rel in gss.split(labeled_idx, y[labeled_idx], labeled_groups):
        splits.append((labeled_idx[tr_rel], labeled_idx[te_rel]))
    return splits


def predict_guard(X_raw, y, fit_idx, eval_idx):
    y_fit = mask_labels(y, eval_idx)
    pipe = model.GUARD().fit(X_raw, y_fit, train_idx=fit_idx)
    return pipe.predict_raw(X_raw[eval_idx])


def predict_sklearn(make_clf, X_raw, y, fit_idx, eval_idx):
    
    scaler = StandardScaler().fit(X_raw[fit_idx])
    clf = make_clf()
    clf.fit(scaler.transform(X_raw[fit_idx]), y[fit_idx])
    return clf.predict(scaler.transform(X_raw[eval_idx]))


def predict_graph(cls, kwargs, X_raw, y, fit_idx, eval_idx, use_unlabeled):
    scaler = StandardScaler().fit(X_raw[fit_idx])
    if use_unlabeled:
        y_train = mask_labels(y, eval_idx)
        g = cls(**kwargs).fit(scaler.transform(X_raw), y_train)
        return g.transduction_[eval_idx]
    subset = np.concatenate([fit_idx, eval_idx])
    y_sub = np.concatenate([y[fit_idx], np.full(len(eval_idx), -1)])
    g = cls(**kwargs).fit(scaler.transform(X_raw[subset]), y_sub)
    return g.transduction_[len(fit_idx):]


# ---------------- Data ----------------

df = model.load_dataset(CSV_PATH)
assert df[LEAF_COL].is_unique
df["hc_neg"] = (df["hc_neg"].astype(str).str.lower().isin(["true", "1"])
                if "hc_neg" in df else False)

y = df[model.LABEL_COL].to_numpy()
hc_neg = df["hc_neg"].to_numpy()
leaves = df[LEAF_COL].to_numpy()
labeled_idx = np.where(y != -1)[0]


def get_embeddings(df):
    if EMB_CACHE and os.path.exists(EMB_CACHE):
        X = np.load(EMB_CACHE)
        if X.shape[0] == len(df):
            print(f"Embeddings: loaded memoized cache {EMB_CACHE}")
            return X
        print(f"[warn] {EMB_CACHE} misaligned ({X.shape[0]} rows vs "
              f"{len(df)}), recomputing")
    embed = model.get_embedder(EMB_MODEL)
    X = embed(df[LEAF_COL].tolist())
    if EMB_CACHE:
        np.save(EMB_CACHE, X)
        print(f"Embeddings: computed and saved to {EMB_CACHE}")
    return X


X_raw = get_embeddings(df)
groups, gdesc = get_groups(df, GROUP_COL, LEAF_COL)

print(f"Embedding model: {EMB_MODEL or model.MODEL_NAME} | shape = {X_raw.shape}")
print(f"Labeled: {len(labeled_idx)} | pos: {(y == 1).sum()} "
      f"| neg: {(y == 0).sum()} | unlabeled: {(y == -1).sum()}")
print(f"Grouping by {gdesc} | sweeps: {N_SWEEPS} | seed offset: {SEED_OFFSET}")


FOLDS = generate_splits(labeled_idx, y, groups, N_SWEEPS, EVAL_FRACTION, SEED_OFFSET)


# ---------------- Evaluation ----------------

def eval_method(predict_fn):
    
    rows, f1s = [], []
    for sweep, (fit_idx, eval_idx) in enumerate(FOLDS):
        y_true = y[eval_idx]
        pred = np.asarray(predict_fn(fit_idx, eval_idx))
        f1, prec, rec = pos_metrics(y_true, pred)
        tp, fp, fn = counts(y_true, pred)

        keep = ~hc_neg[eval_idx]
        f1a, preca, reca = pos_metrics(y_true[keep], pred[keep])

        rows.append({
            "sweep": sweep,
            "seed": SEED_OFFSET + sweep,
            "n_eval": len(eval_idx),
            "n_pos": int((y_true == 1).sum()),
            "f1": f1, "precision": prec, "recall": rec,
            "tp": tp, "fp": fp, "fn": fn,
            "f1_annotated_only": f1a,
            "precision_annotated_only": preca,
            "recall_annotated_only": reca,
        })
        f1s.append(f1)
    return rows, f1s

methods = {}

name_guard = (f"GUARD [ours] (LogReg C={model.GUARD().C}, "
              f"thr={model.DECISION_THRESHOLD})")
methods[name_guard] = lambda tr, te: predict_guard(X_raw, y, tr, te)

cfg_knn, cfg_rf, cfg_lp, cfg_ls = FIXED_KNN, FIXED_RF, FIXED_LP_K, FIXED_LS

methods[f"kNN (k={cfg_knn[0]}, {cfg_knn[1]})"] = \
    lambda tr, te: predict_sklearn(
        lambda: KNeighborsClassifier(n_neighbors=cfg_knn[0],
                                     weights=cfg_knn[1]),
        X_raw, y, tr, te)

methods[f"RF (depth={cfg_rf[0]}, n={cfg_rf[1]}, cw={cfg_rf[2]})"] = \
    lambda tr, te: predict_sklearn(
        lambda: RandomForestClassifier(n_estimators=cfg_rf[1],
                                       max_depth=cfg_rf[0],
                                       class_weight=cfg_rf[2],
                                       random_state=0, n_jobs=-1),
        X_raw, y, tr, te)

methods[f"LabelPropagation (k={cfg_lp}, +unlabeled)"] = \
    lambda tr, te: predict_graph(
        LabelPropagation, {"kernel": "knn", "n_neighbors": cfg_lp,
                           "max_iter": MAX_ITER},
        X_raw, y, tr, te, use_unlabeled=True)

methods[f"LabelPropagation (k={cfg_lp}, labeled-only)"] = \
    lambda tr, te: predict_graph(
        LabelPropagation, {"kernel": "knn", "n_neighbors": cfg_lp,
                           "max_iter": MAX_ITER},
        X_raw, y, tr, te, use_unlabeled=False)

methods[f"LabelSpreading (k={cfg_ls[0]}, a={cfg_ls[1]})"] = \
    lambda tr, te: predict_graph(
        LabelSpreading, {"kernel": "knn", "n_neighbors": cfg_ls[0],
                         "alpha": cfg_ls[1], "max_iter": MAX_ITER},
        X_raw, y, tr, te, use_unlabeled=True)

for name, fname in LLM_CACHES.items():
    pred_map = load_pred_cache(os.path.join(PRED_DIR, fname))
    if pred_map is None:
        print(f"[skip] cache not found: {fname} ({name})")
        continue
    missing = [l for l in leaves[labeled_idx] if l not in pred_map]
    if missing:
        print(f"[warn] {name}: {len(missing)} labeled leaves missing from "
              f"cache (treated as negative)")
    pred_all = np.array([pred_map.get(l, 0) for l in leaves])
    methods[name] = (lambda pa: (lambda tr, te: pa[te]))(pred_all)


per_run_rows, summary_rows, f1_store = [], [], {}

for name, fn in methods.items():
    rows, f1s = eval_method(fn)
    f1_store[name] = f1s
    for r in rows:
        per_run_rows.append({"method": name, "seed_offset": SEED_OFFSET, **r})
        
    a = pd.DataFrame(rows)
    summary_rows.append({
        "method": name,
        "runs": len(rows),
        "f1_mean": a.f1.mean(), "f1_sd": a.f1.std(),
        "precision_mean": a.precision.mean(), "precision_sd": a.precision.std(),
        "recall_mean": a.recall.mean(), "recall_sd": a.recall.std(),
        "f1_min": a.f1.min(), "f1_max": a.f1.max(),
        "tp_mean": a.tp.mean(), "fp_mean": a.fp.mean(), "fn_mean": a.fn.mean(),
        "f1_annotated_only_mean": a.f1_annotated_only.mean(),
        "f1_annotated_only_sd": a.f1_annotated_only.std(),
    })

summary = pd.DataFrame(summary_rows).sort_values("f1_mean", ascending=False)
per_run = pd.DataFrame(per_run_rows)

os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
summary.to_csv(SUMMARY_PATH, index=False)
per_run.to_csv(PER_RUN_PATH, index=False)

print(f"\n===== MEANS over {N_SWEEPS} runs (F1+ desc) =====")
for _, r in summary.iterrows():
    print(f"{r['method']:52s} | F1+ = {r['f1_mean']:.4f} +/- {r['f1_sd']:.4f} "
          f"| prec = {r['precision_mean']:.4f} +/- {r['precision_sd']:.4f} "
          f"| rec = {r['recall_mean']:.4f} +/- {r['recall_sd']:.4f}")

print(f"\n===== PER-RUN, {name_guard} =====")
g = per_run[per_run.method == name_guard]
print(g[["sweep", "seed", "n_eval", "n_pos", "precision", "recall", "f1",
         "tp", "fp", "fn"]].to_string(index=False,
                                      float_format=lambda v: f"{v:.4f}"))

print(f"\nSaved: {SUMMARY_PATH}")
print(f"Saved: {PER_RUN_PATH}  ({len(per_run)} rows: "
      f"{len(methods)} methods x {N_SWEEPS} runs)")