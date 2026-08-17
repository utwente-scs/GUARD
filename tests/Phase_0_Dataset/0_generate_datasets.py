
import argparse
import numpy as np
import pandas as pd

ORACLE = "../../resources/in/ground_truth.csv"
THETA = 0.5


def _normalize_secret(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip().str.lower()
    out = pd.Series(-1, index=s.index, dtype="int8")
    out[s == "true"] = 1
    out[s == "false"] = 0
    return out


def extract_leaf(param_name: str) -> str:
    return str(param_name).strip().split(".")[-1]


def load_oracle(path: str, theta: float, theta_mode: str) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype={"secret": "string"}, low_memory=False)
    raw = raw.dropna(subset=["param_name"])

    occ = pd.DataFrame({
        "leaf": raw["param_name"].map(extract_leaf),
        "y": _normalize_secret(raw["secret"]),
    })
    occ = occ[occ["leaf"] != ""]

    grp = occ.groupby("leaf")["y"]
    agg = pd.DataFrame({
        "n_true":  grp.apply(lambda v: int((v == 1).sum())),
        "n_false": grp.apply(lambda v: int((v == 0).sum())),
        "n_unl":   grp.apply(lambda v: int((v == -1).sum())),
    })
    agg["n_labeled"] = agg["n_true"] + agg["n_false"]
    agg["pos_frac"] = np.where(
        agg["n_labeled"] > 0, agg["n_true"] / agg["n_labeled"].clip(lower=1),
        np.nan)

    agg["label"] = -1
    lab = agg["n_labeled"] > 0
    if theta_mode == "majority":
        agg.loc[lab, "label"] = (agg.loc[lab, "pos_frac"] >= theta).astype("int8")
    elif theta_mode == "any_positive":
        agg.loc[lab, "label"] = (agg.loc[lab, "pos_frac"] > 0).astype("int8")
    else:
        raise ValueError(theta_mode)

    return agg.reset_index()


def promote_hc_negatives(df: pd.DataFrame, min_occ: int) -> pd.DataFrame:
    df = df.copy()
    df["hc_neg"] = False
    mask = (df.label == -1) & (df.n_unl >= min_occ)
    df.loc[mask, ["label", "hc_neg"]] = [0, True]
    return df


def print_summary(df: pd.DataFrame) -> None:
    n_pos = int((df.label == 1).sum())
    n_neg = int((df.label == 0).sum())
    n_hc = int(df.hc_neg.sum())
    n_unl = int((df.label == -1).sum())
    n_lab = n_pos + n_neg
    print(f"unique leaves: {len(df)} | pos: {n_pos} | neg: {n_neg} "
          f"(annotated: {n_neg - n_hc}, heuristic: {n_hc}) | unlabeled: {n_unl}")
    if n_lab:
        print(f"prevalence among labeled: {n_pos / n_lab:.2%}")
    disc = df[(df.n_labeled > 0) & (df.pos_frac > 0) & (df.pos_frac < 1)]
    print(f"leaves with discordant annotations: {len(disc)}")
    print("positives where any_positive differs from majority:", (disc.pos_frac < 0.5).sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default=ORACLE)
    ap.add_argument("--theta", type=float, default=THETA)
    ap.add_argument("--theta-mode", default="majority",
                    choices=["majority", "any_positive"])
    ap.add_argument("--min-occ", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = load_oracle(args.oracle, args.theta, args.theta_mode)
    df = promote_hc_negatives(df, min_occ=args.min_occ)
    print_summary(df)
    df.to_csv(args.out, index=False)
    print(f"Written: {args.out}")