"""EEGLAB epoch features and raw per-subject feature caching."""
import os
import re

import mne
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from .features import extract_epoch_features, make_epoch_feature_names

FEATURE_CACHE_VERSION = 1


def extract_patient_epoch_matrix(set_file_path, epoch_sec=2.0, overlap_sec=0.5):
    """
    Load .set, crop to first 300s, epoch it, and extract per-epoch features.

    Returns:
      X_e: [n_epochs, d_features]
      ch_names: list[str]
    """
    raw = mne.io.read_raw_eeglab(set_file_path, preload=True, verbose=False)
    raw.crop(tmax=300)
    #raw.pick_types(eeg=True)
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


def _source_signature(set_file_path):
    """Signature of the EEGLAB .set file and an optional companion .fdt file."""
    paths = [set_file_path]
    fdt_path = os.path.splitext(set_file_path)[0] + ".fdt"
    if os.path.exists(fdt_path):
        paths.append(fdt_path)

    parts = []
    for path in paths:
        stat = os.stat(path)
        parts.append(f"{os.path.abspath(path)}|{stat.st_size}|{stat.st_mtime_ns}")
    return "||".join(parts)


def load_or_extract_patient_epoch_matrix(
        set_file_path,
        subject_id,
        cache_dir,
        epoch_sec=2.0,
        overlap_sec=0.5,
):
    """
    Load a subject's raw epoch-feature matrix from cache, or compute it once.

    The cache contains only unscaled, unselected features. Standardization and
    F-statistic feature selection remain fold-specific and are fit on inner
    training subjects only.
    """
    os.makedirs(cache_dir, exist_ok=True)
    safe_subject_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(subject_id))
    cache_path = os.path.join(cache_dir, f"{safe_subject_id}.npz")
    signature = _source_signature(set_file_path)

    if os.path.exists(cache_path):
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                valid = (
                    int(cached["cache_version"].item()) == FEATURE_CACHE_VERSION
                    and str(cached["source_signature"].item()) == signature
                    and float(cached["epoch_sec"].item()) == float(epoch_sec)
                    and float(cached["overlap_sec"].item()) == float(overlap_sec)
                )
                if valid:
                    return (
                        cached["X_e"].astype(np.float32, copy=False),
                        cached["ch_names"].astype(str).tolist(),
                    )
        except (OSError, ValueError, KeyError):
            pass

    print(f"[features] Computing and caching subject {subject_id}")
    X_e, ch_names = extract_patient_epoch_matrix(
        set_file_path,
        epoch_sec=epoch_sec,
        overlap_sec=overlap_sec,
    )

    temporary_path = cache_path + ".tmp.npz"
    np.savez(
        temporary_path,
        X_e=np.asarray(X_e, dtype=np.float32),
        ch_names=np.asarray(ch_names, dtype=str),
        cache_version=np.asarray(FEATURE_CACHE_VERSION, dtype=np.int64),
        source_signature=np.asarray(signature),
        epoch_sec=np.asarray(epoch_sec, dtype=np.float64),
        overlap_sec=np.asarray(overlap_sec, dtype=np.float64),
    )
    os.replace(temporary_path, cache_path)
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
        cache_dir=None,
):
    """
    Build epoch-level dataset from subject files.

    Notes:
      - scaler/selector are fit on epochs (train split only), then applied to val/test.
      - subject ids are mapped to compact ints for torch + aggregation.
      - weights are 1 / (#epochs per subject) to reduce subject-imbalance within minibatches.

    Returns:
      ds, selected_feature_names, scaler, selector, ch_names_ref, sid_int, sid_maps
    """
    X_all, y_all, sid_all = [], [], []

    if subject_ids is None:
        subject_ids = [os.path.splitext(os.path.basename(p))[0] for p in file_paths]

    for fp, y, sid in zip(file_paths, labels, subject_ids):
        if cache_dir is None:
            Xe, chn = extract_patient_epoch_matrix(
                fp, epoch_sec=epoch_sec, overlap_sec=overlap_sec
            )
        else:
            Xe, chn = load_or_extract_patient_epoch_matrix(
                fp,
                subject_id=sid,
                cache_dir=cache_dir,
                epoch_sec=epoch_sec,
                overlap_sec=overlap_sec,
            )
        X_all.append(Xe)
        y_all.append(np.full((Xe.shape[0],), int(y), dtype=np.int64))
        sid_all.append(np.full((Xe.shape[0],), str(sid)))

        if ch_names_ref is None:
            ch_names_ref = chn

    X = np.vstack(X_all)              # [N_epochs, d_raw]
    y = np.concatenate(y_all)         # [N_epochs]
    sid_str = np.concatenate(sid_all) # [N_epochs] dtype=str

    # Stable (within this call) mapping: subject string id -> compact int id
    uniq = np.unique(sid_str)
    sid_to_int = {s: i for i, s in enumerate(uniq)}
    int_to_sid = {i: s for s, i in sid_to_int.items()}
    sid_int = np.array([sid_to_int[s] for s in sid_str], dtype=np.int64)

    # Preprocessing on epochs
    if scaler is None:
        scaler, fit_scaler = StandardScaler(), True
    X_scaled = scaler.fit_transform(X) if fit_scaler else scaler.transform(X)

    if selector is not None:
        if fit_selector:
            selector.fit(X_scaled, y)
        X_sel = selector.transform(X_scaled)
    else:
        X_sel = X_scaled

    # Feature names after selection
    full_names = make_epoch_feature_names(ch_names_ref)
    if selector is not None:
        sel_idx = selector.get_support(indices=True)
        selected_names = [full_names[i] for i in sel_idx]
    else:
        selected_names = full_names

    # Per-epoch weights: 1 / (#epochs of its subject)
    uniq_i, cnt = np.unique(sid_int, return_counts=True)
    cnt_map = {u: c for u, c in zip(uniq_i, cnt)}
    weights = np.array([1.0 / cnt_map[s] for s in sid_int], dtype=np.float32)

    ds = EpochDataset(X_sel, y.astype(np.float32), sid_int, weights=weights)
    sid_maps = {"sid_to_int": sid_to_int, "int_to_sid": int_to_sid}
    return ds, selected_names, scaler, selector, ch_names_ref, sid_int, sid_maps
