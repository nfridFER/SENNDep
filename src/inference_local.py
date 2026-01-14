import os
import re
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch

from .data import extract_patient_epoch_matrix
from .eval import aggregate_subject_logits, prob_confidence
from .model import SENN


# ----------------------------
# Concept labels helpers
# ----------------------------
def load_concept_labels_txt(fold_dir: str) -> Dict[int, Dict[str, object]]:
    """
    Loads fold_dir/concept_labels.txt as a concept->(name, features) mapping.
    """
    path = os.path.join(fold_dir, "concept_labels.txt")
    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    header_re = re.compile(
        r"^\s*(?:C\s*)?(?:concept\s*)?(\d{1,3})\s*(?:[:\-]|$)\s*(.*)$",
        re.IGNORECASE,
    )

    out: Dict[int, Dict[str, object]] = {}
    cur: Optional[int] = None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        m = header_re.match(line)
        is_probably_header = False
        if m:
            if re.match(r"^\s*(C|concept)\b", raw, re.IGNORECASE) or re.match(r"^\s*\d+\s*[:\-]", raw):
                is_probably_header = True

        if is_probably_header:
            cid = int(m.group(1))
            name = m.group(2).strip() if m.group(2) else ""
            out[cid] = {"name": name or None, "features": []}
            cur = cid
            continue

        if cur is not None:
            feat = re.sub(r"^\s*[-•\*\u2022]+\s*", "", raw).strip()
            if feat:
                out[cur]["features"].append(feat)

    return out


# ----------------------------
# Concept -> motifs from CSV
# ----------------------------
def load_concept_to_motifs_csv(concept_motif_csv: str, fold_idx: int) -> Dict[int, List[str]]:
    df = pd.read_csv(concept_motif_csv)
    needed = {"Fold", "Concept", "Motif"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{concept_motif_csv} must contain columns {sorted(needed)}. Got: {list(df.columns)}")

    df["Fold"] = df["Fold"].astype(int)
    df["Concept"] = df["Concept"].astype(int)
    df["Motif"] = df["Motif"].astype(str)

    dff = df[df["Fold"] == int(fold_idx)].copy()
    out: Dict[int, List[str]] = {}
    for cid, g in dff.groupby("Concept"):
        out[int(cid)] = g["Motif"].tolist()
    return out


# ----------------------------
# Motif -> group
# ----------------------------
def assign_group(motif_name: str) -> str:
    feature_map = {
        "sampen": "Entropy",
        "renyi_entropy": "Entropy",
        "approx_entropy": "Entropy",
        "shannon_H": "Entropy",
        "higuchi_fd": "Fractal",
        "c0_complexity": "Complexity",
        "relpow_alpha": "AlphaPower",
        "relpow_beta": "BetaPower",
        "relpow_theta": "ThetaPower",
        "relpow_gamma": "GammaPower",
        "relpow_delta": "DeltaPower",
        "std": "Statistical",
        "mean": "Statistical",
        "skew": "Statistical",
        "kurtosis": "Statistical",
        "coh_alpha": "Connectivity",
        "ratio_mean_theta_over_alpha": "Ratio",
        "ratio_mean_theta_over_beta": "Ratio",
        "alpha_asym": "Asymmetry",
    }

    region_map = {
        "T6": "Posterior", "T5": "Posterior", "O1": "Posterior", "O2": "Posterior",
        "F4": "RightFrontal", "F3": "LeftFrontal", "F7": "LeftFrontal", "F8": "RightFrontal",
        "FP1": "LeftFrontal", "FP2": "RightFrontal", "Fp1": "LeftFrontal", "Fp2": "RightFrontal",
        "Fz": "FrontalMid",
        "C3": "Central", "C4": "Central", "Cz": "Central",
        "P3": "Parietal", "P4": "Parietal", "Pz": "Parietal",
        "T4": "Temporal", "T3": "Temporal",
    }

    if motif_name == "coh_alpha_F7-T6":
        return "FrontoTemporal_Connectivity"
    if motif_name == "coh_alpha_T6-O1":
        return "TemporoPosterior_Connectivity"

    m = re.match(r"([a-z0-9_]+)_([A-Za-z0-9\-]+)$", motif_name)
    if not m:
        return "Unknown"

    feat, chan = m.group(1), m.group(2)

    if "-" in chan and feat.startswith("coh_alpha"):
        return "Pair_Connectivity"

    feat_group = feature_map.get(feat, "Other")
    region_group = region_map.get(chan, "UnknownRegion")
    return f"{region_group}_{feat_group}"


def motifs_to_group_summary(motifs: List[str]) -> Dict[str, object]:
    mg = [(m, assign_group(m)) for m in motifs]
    counts = Counter([g for _, g in mg])
    groups_sorted = dict(sorted(counts.items(), key=lambda x: (-x[1], x[0])))
    return {"groups": groups_sorted, "motif_groups": mg}


# ----------------------------
# Fold assets loader
# ----------------------------
def load_fold_assets(fold_dir: str, concept_dim: int, device: Optional[str] = None):
    """
    Loads:
      - scaler.joblib
      - selector.joblib
      - channels.npy
      - best_model.pt
      - concept_labels.npy (optional)
    Returns (model, scaler, selector, ch_ref, concept_labels, device)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    scaler_path = os.path.join(fold_dir, "scaler.joblib")
    selector_path = os.path.join(fold_dir, "selector.joblib")
    channels_path = os.path.join(fold_dir, "channels.npy")
    model_path = os.path.join(fold_dir, "best_model.pt")

    for p, nm in [
        (model_path, "model"),
        (scaler_path, "scaler"),
        (selector_path, "selector"),
        (channels_path, "channels"),
    ]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {nm}: {p}")

    scaler = joblib.load(scaler_path)
    selector = joblib.load(selector_path)
    ch_ref = np.load(channels_path, allow_pickle=True).tolist()

    # infer input_dim from selector support
    input_dim = int(selector.get_support(indices=True).shape[0])

    model = SENN(input_dim=input_dim, concept_dim=concept_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    concept_labels_path = os.path.join(fold_dir, "concept_labels.npy")
    concept_labels = np.load(concept_labels_path, allow_pickle=True).tolist() if os.path.exists(concept_labels_path) else None

    return model, scaler, selector, ch_ref, concept_labels, device


# ----------------------------
# Prediction + local explanation data
# ----------------------------
def predict_eeg_set_local(
        set_path: str,
        fold_dir: str,
        fold_idx: int,
        concept_dim: int = 50,
        epoch_sec: float = 2.0,
        overlap_sec: float = 0.5,
        agg_method: str = "trimmed",
        trim_p: float = 0.1,
        confidence_weighting: bool = False,
        top_concepts: int = 5,
        concept_motif_csv: Optional[str] = None,
) -> Dict[str, object]:
    model, scaler, selector, ch_ref, concept_labels, device = load_fold_assets(
        fold_dir=fold_dir, concept_dim=concept_dim
    )

    concept_txt = load_concept_labels_txt(fold_dir)

    concept_to_motifs: Dict[int, List[str]] = {}
    if concept_motif_csv and os.path.exists(concept_motif_csv):
        concept_to_motifs = load_concept_to_motifs_csv(concept_motif_csv, fold_idx=fold_idx)

    # --- features (REUSE training extractor) ---
    X_raw, ch_names = extract_patient_epoch_matrix(set_path, epoch_sec=epoch_sec, overlap_sec=overlap_sec)

    n_ch = len(ch_ref)
    expected_full_dim = (5 * n_ch) + (10 * n_ch) + 2 + (n_ch * (n_ch - 1)) // 2 + 1
    if X_raw.shape[1] != expected_full_dim:
        raise ValueError(
            "Raw feature dimension mismatch.\n"
            f"  got: {X_raw.shape[1]}\n"
            f"  expected (from fold channels): {expected_full_dim}\n"
            f"  fold channels n={len(ch_ref)}  file channels n={len(ch_names)}\n"
            "Fix: use a .set with the same channel layout as training (for this demo)."
        )

    X_scaled = scaler.transform(X_raw)
    X_sel = selector.transform(X_scaled).astype(np.float32)

    X_t = torch.as_tensor(X_sel, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits, theta, z = model(X_t)
        logits = logits.squeeze(-1).cpu().numpy()
        theta = theta.cpu().numpy()
        z = z.cpu().numpy()

    # single subject aggregation
    sids = np.zeros((len(logits),), dtype=np.int64)
    subj_probs = aggregate_subject_logits(
        logits, sids, method=agg_method, trim_p=trim_p, confidence_weighting=confidence_weighting
    )
    p_mdd = float(subj_probs[0])
    conf = prob_confidence(p_mdd)

    # concept contributions = theta*z, then trimmed mean across epochs
    contrib = theta * z  # [n_epochs, concept_dim]
    from scipy.stats import trim_mean
    agg_contrib = np.apply_along_axis(lambda v: trim_mean(v, proportiontocut=trim_p / 2), 0, contrib)

    top_idx = np.argsort(-np.abs(agg_contrib))[:top_concepts]

    top = []
    for ci in top_idx:
        rows = []
        if concept_labels is not None and ci < len(concept_labels):
            for (fname, sgn, rabs) in concept_labels[ci]:
                rows.append({"feature": str(fname), "sign": "+" if float(sgn) >= 0 else "-", "abs_r": float(rabs)})

        txt_info = concept_txt.get(int(ci), {})
        concept_name = txt_info.get("name")
        txt_features = txt_info.get("features", [])

        motifs = concept_to_motifs.get(int(ci), [])
        motif_info = motifs_to_group_summary(motifs) if motifs else {"groups": {}, "motif_groups": []}

        top.append(
            {
                "concept": int(ci),
                "concept_name": concept_name,
                "concept_txt_features": txt_features,
                "contribution": float(agg_contrib[ci]),
                "features": rows,
                "motifs": motifs,
                "motif_groups": motif_info["motif_groups"],
                "motif_group_counts": motif_info["groups"],
            }
        )

    return {
        "set_path": set_path,
        "fold_dir": fold_dir,
        "fold_idx": int(fold_idx),
        "concept_motif_csv": concept_motif_csv,
        "p_mdd": p_mdd,
        "confidence": conf,
        "pred_label": "MDD" if p_mdd >= 0.5 else "Healthy",
        "top_concepts": top,
        "n_epochs": int(len(logits)),
        "epoch_sec": float(epoch_sec),
        "overlap_sec": float(overlap_sec),
    }


# ----------------------------
# Explanation tables (reuse motif artifacts if present)
# ----------------------------
def load_resources(interpret_dir: str):
    """
    Optional CSVs in interpret_dir:
      - concept_to_motifs_full.csv (required if you want CSV motif mapping)
      - motif_group_mapping.csv    (optional; preferred over heuristic)
      - motifs_mdd_grouped.csv     (optional; global stats)
      - motifs_healthy_grouped.csv (optional; global stats)
    """
    concept_map_path = os.path.join(interpret_dir, "concept_to_motifs_full.csv")
    if not os.path.exists(concept_map_path):
        raise FileNotFoundError(f"Missing: {concept_map_path}")

    concept_map = pd.read_csv(concept_map_path)
    concept_map["Fold"] = concept_map["Fold"].astype(int)
    concept_map["Concept"] = concept_map["Concept"].astype(int)
    concept_map["Motif"] = concept_map["Motif"].astype(str)

    motif_group_map_path = os.path.join(interpret_dir, "motif_group_mapping.csv")
    motif_group_map = pd.read_csv(motif_group_map_path) if os.path.exists(motif_group_map_path) else None
    if motif_group_map is not None:
        motif_group_map["Motif"] = motif_group_map["Motif"].astype(str)
        motif_group_map["Group"] = motif_group_map["Group"].astype(str)

    mdd_path = os.path.join(interpret_dir, "motifs_mdd_grouped.csv")
    healthy_path = os.path.join(interpret_dir, "motifs_healthy_grouped.csv")
    mdd_groups = pd.read_csv(mdd_path) if os.path.exists(mdd_path) else None
    healthy_groups = pd.read_csv(healthy_path) if os.path.exists(healthy_path) else None

    return concept_map, motif_group_map, mdd_groups, healthy_groups


def explain_prediction(out: Dict[str, object], fold_number: int, interpret_dir: str, prefer_csv_grouping: bool = True) -> pd.DataFrame:
    concept_map, motif_group_map, mdd_groups, healthy_groups = load_resources(interpret_dir)

    p_val = float(out["p_mdd"])
    group_table = None
    if (mdd_groups is not None) and (healthy_groups is not None):
        group_table = mdd_groups if p_val > 0.5 else healthy_groups

    concepts = [(int(c["concept"]), float(c["contribution"])) for c in out.get("top_concepts", [])]
    results = []

    for c_id, contrib in concepts:
        motifs = concept_map[
            (concept_map["Fold"] == int(fold_number)) &
            (concept_map["Concept"] == int(c_id))
            ]["Motif"].astype(str).tolist()

        if len(motifs) == 0:
            results.append({
                "Concept": c_id,
                "Contribution": contrib,
                "Motif": "NOT FOUND",
                "Group": "NOT FOUND",
                "Group total abs contrib": None,
                "Group mean signed contrib": None,
            })
            continue

        for motif in motifs:
            if prefer_csv_grouping and motif_group_map is not None:
                group_row = motif_group_map[motif_group_map["Motif"] == motif]
                group_name = group_row["Group"].iloc[0] if len(group_row) else "Unknown"
            else:
                group_name = assign_group(motif)

            total_abs = None
            mean_signed = None
            if group_table is not None and "Group" in group_table.columns:
                gt = group_table.copy()
                gt["Group"] = gt["Group"].astype(str)
                if str(group_name) in set(gt["Group"].values):
                    g = gt[gt["Group"] == str(group_name)].iloc[0]
                    total_abs = g.get("Total abs contrib", None)
                    mean_signed = g.get("Mean signed contrib", None)

            results.append({
                "Concept": c_id,
                "Contribution": contrib,
                "Motif": motif,
                "Group": str(group_name),
                "Group total abs contrib": total_abs,
                "Group mean signed contrib": mean_signed,
            })

    return pd.DataFrame(results)


def summarize_subject_by_group(df_local_expl: pd.DataFrame, min_contribution: float = 1.0, top_frac: Optional[float] = None) -> pd.DataFrame:
    df = df_local_expl.copy()

    grouped = (
        df.groupby("Group")
        .agg(
            Total_Local_Contribution=("Contribution", "sum"),
            Concept_Count=("Concept", "nunique"),
            Global_Mean_Signed=("Group mean signed contrib", "first"),
            Global_Total_Abs=("Group total abs contrib", "first"),
        )
        .sort_values("Total_Local_Contribution", ascending=False)
        .reset_index()
    )

    if top_frac is not None and not grouped.empty:
        grouped["Cumulative"] = grouped["Total_Local_Contribution"].cumsum()
        total = grouped["Total_Local_Contribution"].sum() or 1.0
        grouped["Cumulative_frac"] = grouped["Cumulative"] / total
        grouped = grouped[grouped["Cumulative_frac"] <= top_frac]

    grouped = grouped[grouped["Total_Local_Contribution"] >= min_contribution]
    if not grouped.empty:
        total_kept = grouped["Total_Local_Contribution"].sum() or 1.0
        grouped["Contribution_frac"] = grouped["Total_Local_Contribution"] / total_kept

    grouped = grouped.drop(columns=["Total_Local_Contribution", "Global_Mean_Signed", "Global_Total_Abs"], errors="ignore")
    grouped = grouped.drop(columns=["Cumulative", "Cumulative_frac"], errors="ignore")
    return grouped
    
    
    
# ----------------------------
# Orchestration 
# ----------------------------
def run_local_inference(
    cfg,
    set_path: str,
    fold_idx: int,
    *,
    prefer_csv_grouping: bool = True,
    save_group_summary_csv: bool = True,
    group_summary_filename: str = "subject_group_summary.csv",
) -> Dict[str, object]:
    """
    Thin orchestration wrapper that:
      - resolves fold_dir + interpret_dir
      - runs predict_eeg_set_local
      - (optionally) builds df_raw + df_summary if motif CSV resources exist
      - (optionally) saves df_summary

    Returns a dict with keys:
      out, df_raw, df_summary, paths, warnings
    """
    fold_dir = os.path.join(cfg.save_dir, f"fold_{int(fold_idx)}")
    if not os.path.isdir(fold_dir):
        raise FileNotFoundError(f"Fold directory not found: {fold_dir}")

    interpret_dir = os.path.join(cfg.save_dir, cfg.infer_interpret_subdir)
    concept_motif_csv = os.path.join(interpret_dir, "concept_to_motifs_full.csv")

    warnings: List[str] = []
    if not os.path.exists(concept_motif_csv):
        warnings.append(f"Missing {concept_motif_csv}. Will skip group-level explanation tables.")

    out = predict_eeg_set_local(
        set_path=set_path,
        fold_dir=fold_dir,
        fold_idx=int(fold_idx),
        concept_dim=cfg.concept_dim,
        epoch_sec=cfg.infer_epoch_sec,
        overlap_sec=cfg.infer_overlap_sec,
        agg_method=cfg.infer_agg_method,
        trim_p=cfg.infer_trim_p,
        confidence_weighting=cfg.infer_confidence_weighting,
        top_concepts=cfg.infer_top_concepts,
        concept_motif_csv=concept_motif_csv if os.path.exists(concept_motif_csv) else None,
    )

    df_raw = None
    df_summary = None
    saved_csv = None

    if os.path.exists(concept_motif_csv):
        df_raw = explain_prediction(
            out,
            fold_number=int(fold_idx),
            interpret_dir=interpret_dir,
            prefer_csv_grouping=prefer_csv_grouping,
        )
        df_summary = summarize_subject_by_group(df_raw, min_contribution=1.0, top_frac=0.9)

        if save_group_summary_csv:
            os.makedirs(interpret_dir, exist_ok=True)
            saved_csv = os.path.join(interpret_dir, group_summary_filename)
            df_summary.to_csv(saved_csv, index=False)

    return {
        "out": out,
        "df_raw": df_raw,
        "df_summary": df_summary,
        "paths": {
            "fold_dir": fold_dir,
            "interpret_dir": interpret_dir,
            "concept_motif_csv": concept_motif_csv,
            "saved_group_summary_csv": saved_csv,
        },
        "warnings": warnings,
    }


