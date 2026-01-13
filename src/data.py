import numpy as np
import mne
import os
import torch
from sklearn.preprocessing import StandardScaler
import re
import pandas as pd

from .features import extract_epoch_features, make_epoch_feature_names


def extract_patient_epoch_matrix(set_file_path, epoch_sec=2.0, overlap_sec=0.5):
    """
    Load .set, crop to first 300s, epoch it, and extract per-epoch features.

    Returns:
      X_e: [n_epochs, d_features]
      ch_names: list[str]
    """
    raw = mne.io.read_raw_eeglab(set_file_path, preload=True, verbose=False)
    raw.crop(tmax=300)
    raw.pick(picks="eeg")

    sfreq = raw.info["sfreq"]
    epochs = mne.make_fixed_length_epochs(
        raw, duration=epoch_sec, overlap=overlap_sec, preload=True, verbose=False
    )

    epoch_data = epochs.get_data()  # [n_epochs, n_channels, n_times]
    ch_names = raw.ch_names

    X_e = np.asarray(
        [extract_epoch_features(ep, sfreq, ch_names) for ep in epoch_data],
        dtype=np.float32,
    )
    return X_e, ch_names





class EpochDataset(torch.utils.data.Dataset):
    """Epoch-level dataset; each epoch inherits the subject label."""

    def __init__(self, X, y, subject_ids, weights=None):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y.reshape(-1, 1), dtype=torch.float32)
        self.sid = torch.as_tensor(subject_ids, dtype=torch.int64)
        self.w = None if weights is None else torch.as_tensor(weights, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        if self.w is None:
            return self.X[i], self.y[i], self.sid[i]
        return self.X[i], self.y[i], self.sid[i], self.w[i]


def build_epoch_dataset(
        file_paths,
        labels,
        selector=None,
        fit_selector=False,
        scaler=None,
        fit_scaler=False,
        epoch_sec=2.0,
        overlap_sec=0.5,
        subject_ids=None,
        ch_names_ref=None,
):
    """
    Returns:
      ds, selected_feature_names, scaler, selector, ch_names_ref, sid_int, sid_maps
    """
    X_all, y_all, sid_all = [], [], []

    if subject_ids is None:
        subject_ids = [os.path.splitext(os.path.basename(p))[0] for p in file_paths]

    for fp, y, sid in zip(file_paths, labels, subject_ids):
        Xe, chn = extract_patient_epoch_matrix(fp, epoch_sec=epoch_sec, overlap_sec=overlap_sec)
        X_all.append(Xe)
        y_all.append(np.full((Xe.shape[0],), int(y), dtype=np.int64))
        sid_all.append(np.full((Xe.shape[0],), str(sid)))

        if ch_names_ref is None:
            ch_names_ref = chn

    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    sid_str = np.concatenate(sid_all)

    uniq = np.unique(sid_str)
    sid_to_int = {s: i for i, s in enumerate(uniq)}
    int_to_sid = {i: s for s, i in sid_to_int.items()}
    sid_int = np.array([sid_to_int[s] for s in sid_str], dtype=np.int64)

    if scaler is None:
        scaler, fit_scaler = StandardScaler(), True
    X_scaled = scaler.fit_transform(X) if fit_scaler else scaler.transform(X)

    if selector is not None:
        if fit_selector:
            selector.fit(X_scaled, y)
        X_sel = selector.transform(X_scaled)
    else:
        X_sel = X_scaled

    full_names = make_epoch_feature_names(ch_names_ref)
    if selector is not None:
        sel_idx = selector.get_support(indices=True)
        selected_names = [full_names[i] for i in sel_idx]
    else:
        selected_names = full_names

    uniq_i, cnt = np.unique(sid_int, return_counts=True)
    cnt_map = {u: c for u, c in zip(uniq_i, cnt)}
    weights = np.array([1.0 / cnt_map[s] for s in sid_int], dtype=np.float32)

    ds = EpochDataset(X_sel, y.astype(np.float32), sid_int, weights=weights)
    sid_maps = {"sid_to_int": sid_to_int, "int_to_sid": int_to_sid}
    return ds, selected_names, scaler, selector, ch_names_ref, sid_int, sid_maps



def load_labels(excel_path: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path)
    df["REC_ID"] = df["REC_ID"].astype(str)
    df["label"] = df["DIAGNOSIS"].apply(lambda d: 0 if str(d).lower().strip() == "healthy" else 1)
    df["is_GA"] = df["REC_ID"].str.startswith("GA")
    return df


def build_id_to_file_and_label(data_folder: str, df: pd.DataFrame):
    id_to_file = {}
    id_to_label = {}

    for fname in os.listdir(data_folder):
        if not fname.endswith(".set"):
            continue
        stem = os.path.splitext(fname)[0]
        cleaned = re.sub(r"^(EEG_|rawEEG_|EEGbefore_|EEGbefore)", "", stem, flags=re.IGNORECASE)

        if cleaned in df["REC_ID"].values:
            id_to_file[cleaned] = os.path.join(data_folder, fname)
            id_to_label[cleaned] = int(df.loc[df["REC_ID"] == cleaned, "label"].values[0])

    return id_to_file, id_to_label


def make_balanced_fold_val_ids(df: pd.DataFrame, n_folds: int, seed: int):
    ga_healthy = df[(df["is_GA"]) & (df["label"] == 0)].reset_index(drop=True)
    old_healthy = df[(~df["is_GA"]) & (df["label"] == 0)].reset_index(drop=True)
    old_mdd = df[(~df["is_GA"]) & (df["label"] == 1)].reset_index(drop=True)

    mdd_ids = old_mdd["REC_ID"].tolist()
    old_h_ids = old_healthy["REC_ID"].tolist()
    ga_h_ids = ga_healthy["REC_ID"].tolist()

    rng = np.random.default_rng(seed)
    rng.shuffle(mdd_ids)
    rng.shuffle(old_h_ids)
    rng.shuffle(ga_h_ids)

    def chunk(lst, n):
        return np.array_split(np.array(lst, dtype=object), n)

    mdd_chunks = chunk(mdd_ids, n_folds)
    old_h_chunks = chunk(old_h_ids, n_folds)
    ga_h_chunks = chunk(ga_h_ids, n_folds)

    fold_val_ids = []
    for fold_idx in range(n_folds):
        mdd_fold = list(mdd_chunks[fold_idx])
        old_fold = list(old_h_chunks[fold_idx])
        ga_fold = list(ga_h_chunks[fold_idx])

        healthy_pool = old_fold + ga_fold
        target = min(len(mdd_fold), len(healthy_pool))

        desired_ga = min(len(ga_fold), target // 2)
        desired_old = target - desired_ga
        if len(old_fold) < desired_old:
            extra = desired_old - len(old_fold)
            take_more_ga = min(extra, len(ga_fold) - desired_ga)
            desired_ga += take_more_ga
            desired_old = len(old_fold)

        val_ids = mdd_fold[:target] + old_fold[:desired_old] + ga_fold[:desired_ga]
        fold_val_ids.append(val_ids)

    return fold_val_ids


def print_fold_stats(df: pd.DataFrame, fold_val_ids):
    for i, ids in enumerate(fold_val_ids):
        sub = df[df["REC_ID"].isin(ids)]
        print(f"\nFold {i} VAL: n={len(ids)}")
        print("  healthy:", (sub["label"] == 0).sum(),
              "  MDD:", (sub["label"] == 1).sum(),
              "  GA:", sub["is_GA"].sum())
        print("    ├─ old healthy:", ((sub["label"] == 0) & (~sub["is_GA"])).sum())
        print("    └─ GA healthy: ", ((sub["label"] == 0) & (sub["is_GA"])).sum())
