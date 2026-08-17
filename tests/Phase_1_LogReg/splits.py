
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

HOLDOUT_FRACTION = 0.30
PARTITION_SEED = 7

GROUP_COL = "project"


def get_groups(df, group_col=GROUP_COL):
    if group_col is None or group_col not in df.columns:
        return None
    return df[group_col].to_numpy()


def make_partition(df, y, group_col=GROUP_COL):
    labeled = np.where(y != -1)[0]
    groups = get_groups(df, group_col)

    if groups is None:
        print("[partition] no group column -> row-level split "
              "(cross-project generalisation is NOT measured)")
        select_idx, holdout_idx = train_test_split(
            labeled,
            test_size=HOLDOUT_FRACTION,
            random_state=PARTITION_SEED,
            stratify=y[labeled],
        )
    else:
        print(f"[partition] grouping on '{group_col}' -> project-disjoint split")
        gss = GroupShuffleSplit(
            n_splits=1, test_size=HOLDOUT_FRACTION, random_state=PARTITION_SEED
        )
        a, b = next(gss.split(labeled, y[labeled], groups=groups[labeled]))
        select_idx, holdout_idx = labeled[a], labeled[b]

    assert not (set(select_idx) & set(holdout_idx)), "select/holdout overlap"
    print(f"[partition] select={len(select_idx)}  holdout={len(holdout_idx)}  "
          f"pos_rate_select={y[select_idx].mean():.3f}  "
          f"pos_rate_holdout={y[holdout_idx].mean():.3f}")
    return np.sort(select_idx), np.sort(holdout_idx)


def mask_labels(y, hidden_idx):
    
    y_masked = y.copy()
    y_masked[hidden_idx] = -1
    return y_masked


def inner_splits(select_idx, y, groups, n_sweeps, val_fraction, seed_offset=0):

    for seed in range(seed_offset, seed_offset + n_sweeps):
        if groups is None:
            rng = np.random.default_rng(seed)
            n_val = max(1, int(len(select_idx) * val_fraction))
            val_idx = rng.choice(select_idx, size=n_val, replace=False)
            fit_idx = np.setdiff1d(select_idx, val_idx)
        else:
            gss = GroupShuffleSplit(
                n_splits=1, test_size=val_fraction, random_state=seed
            )
            a, b = next(
                gss.split(select_idx, y[select_idx], groups=groups[select_idx])
            )
            fit_idx, val_idx = select_idx[a], select_idx[b]
        yield fit_idx, val_idx