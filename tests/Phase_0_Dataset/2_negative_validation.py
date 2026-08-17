
import pandas as pd
from scipy import stats

CSV_PATH = "../../resources/datasets/dataset_t3.csv"
LEAF_COL = "leaf"
SAMPLE_SIZE = 200
SEED = 42
REVIEW_CSV = "../../resources/out/negatives_review.csv"
MODE = "report"          # "sample" | "report"

if MODE == "sample":
    df = pd.read_csv(CSV_PATH, dtype={LEAF_COL: str}, keep_default_na=False)
    pool = df[(df["label"] == 0) & (df["hc_neg"] == True)]
    print(f"Heuristic negatives available: {len(pool)}")

    sample = pool.sample(min(SAMPLE_SIZE, len(pool)), random_state=SEED)
    sample = sample[[LEAF_COL, "n_unl"]].rename(
        columns={"n_unl": "occurrences"}).sort_values("occurrences")
    sample["manual_label"] = ""
    sample.to_csv(REVIEW_CSV, index=False)
    print(f"Sampled {len(sample)} -> {REVIEW_CSV}")
    print("Fill 'manual_label' with 0 (not a secret) / 1 (secret), "
          "then rerun with MODE='report'.")

elif MODE == "report":
    rev = pd.read_csv(REVIEW_CSV, dtype={LEAF_COL: str})
    rev = rev[rev["manual_label"].astype(str).str.strip().isin(["0", "1"])]
    labels = rev["manual_label"].astype(int)
    n, errors = len(labels), int(labels.sum())
    rate = errors / n
    lo, hi = stats.beta.interval(0.95, errors + 0.5, n - errors + 0.5)
    print(f"Reviewed: {n} | heuristic errors (true secrets): {errors}")
    print(f"Error rate: {rate:.3f} (95% CI [{lo:.3f}, {hi:.3f}])")
    wrong = rev[labels == 1]
    if len(wrong):
        print("\nMislabeled leaves (occurrences):")
        for _, r in wrong.iterrows():
            print(f"  {r[LEAF_COL]} ({r['occurrences']})")