import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mne


# ----------------------------
# Parsing regex
# ----------------------------
_CONCEPT_HEADER_RE = re.compile(r"^Concept\s+(\d+):")
_FEATURE_LINE_RE = re.compile(r"([a-z0-9_]+)\[([A-Z0-9\-]+)\].*\|r\|=([0-9.]+)")
_PROB_RE = re.compile(r"p\(MDD\)\s*=\s*([0-9.eE+-]+)")
_CONTRIB_RE = re.compile(r"^C(\d+):\s*([+-]?[0-9.]+)")
_SUBJECT_RE = re.compile(r"^Subject\s+(\S+)")


# ----------------------------
# Topomap setup
# ----------------------------
ALL_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz",
    "C3", "C4", "Cz", "P3", "P4", "Pz",
    "O1", "O2", "T3", "T4", "T5", "T6",
]

_montage = mne.channels.make_standard_montage("standard_1020")
_info = mne.create_info(ch_names=ALL_CHANNELS, sfreq=1000.0, ch_types="eeg")
_info.set_montage(_montage)


# ----------------------------
# Helpers
# ----------------------------
def _try_import_seaborn():
    try:
        import seaborn as sns  # noqa
        return sns
    except Exception:
        return None


def _keep_top_correlated(features: List[Tuple[str, str, float]], keep_frac: float = 0.7) -> List[str]:
    """
    features: list[(ftype, loc, r)]
    Keeps all with r >= keep_frac * max_r.
    Returns motif tokens like f"{ftype}_{loc}".
    """
    if not features:
        return []
    features = sorted(features, key=lambda x: x[2], reverse=True)
    max_r = features[0][2]
    return [f"{ftype}_{loc}" for ftype, loc, r in features if r >= keep_frac * max_r]


def parse_concepts(concept_file: Path, keep_frac: float = 0.7) -> Dict[int, List[str]]:
    """
    Parse concept_labels.txt into concept_id -> [motif_name,...]
    """
    concept_to_motifs: Dict[int, List[str]] = {}

    current_id: Optional[int] = None
    feats: List[Tuple[str, str, float]] = []

    with concept_file.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()

            m = _CONCEPT_HEADER_RE.match(line)
            if m:
                if current_id is not None:
                    concept_to_motifs[current_id] = _keep_top_correlated(feats, keep_frac=keep_frac)
                current_id = int(m.group(1))
                feats = []
                continue

            if current_id is None:
                continue

            if line.startswith(("+", "-")):
                fm = _FEATURE_LINE_RE.search(line)
                if fm:
                    feats.append((fm.group(1), fm.group(2), float(fm.group(3))))

    if current_id is not None:
        concept_to_motifs[current_id] = _keep_top_correlated(feats, keep_frac=keep_frac)

    return concept_to_motifs


def parse_subjects(explanation_file: Path, concept_to_motifs: Dict[int, List[str]]):
    """
    Parse subject_explanations.txt and distribute concept contributions across motifs.
    Returns:
      mdd_contribs:    list[dict[motif->value]]
      healthy_contribs:list[dict[motif->value]]
    """
    mdd_contribs = []
    healthy_contribs = []

    subj_id = None
    group = None
    contribs = defaultdict(float)

    def _flush():
        nonlocal subj_id, group, contribs
        if subj_id is None or group is None:
            return
        target = mdd_contribs if group == "MDD" else healthy_contribs
        target.append(dict(contribs))

    with explanation_file.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()

            sm = _SUBJECT_RE.match(line)
            if sm:
                _flush()
                subj_id = sm.group(1)
                group = None
                contribs = defaultdict(float)
                continue

            pm = _PROB_RE.search(line)
            if pm:
                prob = float(pm.group(1))
                group = "MDD" if prob > 0.5 else "Healthy"
                continue

            cm = _CONTRIB_RE.match(line)
            if cm:
                c_id = int(cm.group(1))
                contrib = float(cm.group(2))
                motifs = concept_to_motifs.get(c_id, [f"unknown_C{c_id}"])
                w = contrib / max(len(motifs), 1)
                for motif in motifs:
                    contribs[motif] += w

    _flush()
    return mdd_contribs, healthy_contribs


def summarize_motifs(contrib_list: List[Dict[str, float]]) -> pd.DataFrame:
    all_motifs = defaultdict(list)
    for subj_dict in contrib_list:
        for motif, val in subj_dict.items():
            all_motifs[motif].append(val)

    rows = []
    for motif, vals in all_motifs.items():
        pos = sum(v > 0 for v in vals)
        neg = sum(v < 0 for v in vals)
        rows.append(
            {
                "Motif": motif,
                "Mean signed contrib": float(np.mean(vals)),
                "Total abs contrib": float(np.sum(np.abs(vals))),
                "Polarity": f"{pos} / {neg}",
                "N": len(vals),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Total abs contrib", ascending=False).reset_index(drop=True)
    return df


def split_polarity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[["Pos", "Neg"]] = df["Polarity"].str.split(" / ", expand=True).astype(int)
    return df


def assign_group(motif_name: str) -> str:
    feature_map = {
        "sampen": "Entropy",
        "renyi_entropy": "Entropy",
        "higuchi_fd": "Fractal",
        "c0_complexity": "Complexity",
        "relpow_alpha": "AlphaPower",
        "relpow_beta": "BetaPower",
        "relpow_theta": "ThetaPower",
        "relpow_gamma": "GammaPower",
        "std": "Statistical",
        "coh_alpha": "Connectivity",
    }

    region_map = {
        "T6": "Posterior",
        "T5": "Posterior",
        "O1": "Posterior",
        "O2": "Posterior",
        "F4": "RightFrontal",
        "F3": "LeftFrontal",
        "F7": "LeftFrontal",
        "F8": "RightFrontal",
        "Fp1": "LeftFrontal",
        "Fp2": "RightFrontal",
        "Fz": "FrontalMid",
        "C3": "Central",
        "C4": "Central",
        "Cz": "Central",
        "P3": "Parietal",
        "P4": "Parietal",
        "Pz": "Parietal",
        "T4": "Temporal",
        "T3": "Temporal",
    }

    if motif_name == "coh_alpha_F7-T6":
        return "FrontoTemporal_Connectivity"
    if motif_name == "coh_alpha_T6-O1":
        return "TemporoPosterior_Connectivity"

    m = re.match(r"([a-z0-9_]+)_([A-Z0-9\-]+)", motif_name)
    if not m:
        return "Unknown"

    feat, chan = m.group(1), m.group(2)
    feat_group = feature_map.get(feat, "Other")
    region_group = region_map.get(chan, "UnknownRegion")
    return f"{region_group}_{feat_group}"


def summarize_grouped(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Group"] = df["Motif"].apply(assign_group)

    def _polarity_sum(series):
        pos = 0
        neg = 0
        for p in series:
            a, b = p.split(" / ")
            pos += int(a)
            neg += int(b)
        return f"{pos} / {neg}"

    grouped = (
        df.groupby("Group", as_index=False)
        .agg(
            {
                "Mean signed contrib": "mean",
                "Total abs contrib": "sum",
                "Polarity": _polarity_sum,
                "N": "sum",
            }
        )
        .sort_values("Total abs contrib", ascending=False)
        .reset_index(drop=True)
    )
    return grouped


def _savefig_pdf(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, format="pdf")
    plt.close()


def plot_top_groups_bar(df_grouped: pd.DataFrame, title: str, out_pdf: Path, top_n: int = 10):
    sns = _try_import_seaborn()
    top = df_grouped.sort_values("Total abs contrib", ascending=False).head(top_n)

    plt.figure(figsize=(12, 6))
    if sns is not None:
        sns.barplot(data=top, x="Total abs contrib", y="Group")
    else:
        plt.barh(top["Group"], top["Total abs contrib"])
        plt.gca().invert_yaxis()

    plt.title(title)
    plt.xlabel("Total Absolute Contribution")
    plt.ylabel("Motif Group")
    _savefig_pdf(out_pdf)


def plot_polarity_scatter(df_grouped: pd.DataFrame, title: str, out_pdf: Path):
    sns = _try_import_seaborn()
    dfp = split_polarity(df_grouped)

    plt.figure(figsize=(12, 6))
    if sns is not None:
        ax = sns.scatterplot(
            data=dfp,
            x="Pos",
            y="Neg",
            size="Total abs contrib",
            hue="Total abs contrib",
            sizes=(20, 300),
            legend=False,
        )
    else:
        ax = plt.gca()
        ax.scatter(dfp["Pos"], dfp["Neg"], s=50)

    for _, row in dfp.iterrows():
        ax.text(row["Pos"] + 0.5, row["Neg"] + 0.3, row["Group"], fontsize=8)

    plt.title(title)
    plt.xlabel("# Subjects with Positive Contribution")
    plt.ylabel("# Subjects with Negative Contribution")
    plt.grid(True)
    _savefig_pdf(out_pdf)


def plot_tug_of_war(df_mdd_grouped: pd.DataFrame, df_healthy_grouped: pd.DataFrame, out_pdf: Path, top_k: int = 15):
    df_tw = pd.merge(
        df_mdd_grouped[["Group", "Mean signed contrib"]].rename(columns={"Mean signed contrib": "MDD"}),
        df_healthy_grouped[["Group", "Mean signed contrib"]].rename(columns={"Mean signed contrib": "Healthy"}),
        on="Group",
        how="inner",
    )
    df_tw["Delta"] = df_tw["MDD"] - df_tw["Healthy"]
    df_tw["Abs Delta"] = df_tw["Delta"].abs()
    df_tw = df_tw.sort_values("Abs Delta", ascending=False).head(top_k).sort_values("Delta")

    fig, ax = plt.subplots(figsize=(12, max(6, 0.4 * len(df_tw))))
    y = np.arange(len(df_tw))

    ax.barh(y, df_tw["Healthy"], label="Healthy")
    ax.barh(y, df_tw["MDD"], alpha=0.6, label="MDD")
    ax.axvline(0, color="black", lw=1)

    ax.set_yticks(y)
    ax.set_yticklabels(df_tw["Group"])
    ax.set_xlabel("Mean Signed Contribution")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)


def extract_channel(motif_name: str) -> str:
    m = re.match(r"[a-z0-9_]+_([A-Z0-9\-]+)", motif_name)
    return m.group(1) if m else "Unknown"


def channel_importance(df: pd.DataFrame) -> Dict[str, float]:
    ch_counter = defaultdict(float)
    for _, row in df.iterrows():
        ch = extract_channel(row["Motif"])
        ch_counter[ch] += float(row["Total abs contrib"])
    return dict(sorted(ch_counter.items(), key=lambda x: x[1], reverse=True))


def plot_channel_contrib(ch_dict: Dict[str, float], title: str, out_pdf: Path):
    channels = list(ch_dict.keys())
    values = list(ch_dict.values())

    plt.figure(figsize=(10, 6))
    plt.bar(channels, values)
    plt.title(title)
    plt.ylabel("Total Absolute Contribution")
    plt.xticks(rotation=45)
    _savefig_pdf(out_pdf)


def build_topomap_data(ch_contrib: Dict[str, float], all_chs: List[str]) -> np.ndarray:
    return np.array([ch_contrib.get(ch, 0.0) for ch in all_chs], dtype=float)


def plot_topomap_contrib(data_vec: np.ndarray, title: str, out_pdf: Path, vmax: float, cmap: str):
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="white")
    ax.set_facecolor("white")

    im, _ = mne.viz.plot_topomap(
        data_vec,
        _info,
        axes=ax,
        show=False,
        cmap=cmap,
        contours=0,
        outlines="head",
    )

    im.set_clim(0, vmax)
    cbar = plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.6)
    cbar.set_label("Relative relevance (a.u.)")

    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf", facecolor="white")
    plt.close(fig)


# ----------------------------
# Public entrypoint
# ----------------------------
def run_motif_analysis(cfg, root_dir: Optional[str] = None) -> Dict[str, object]:
    """
    Reads fold_*/concept_labels.txt + fold_*/subject_explanations.txt and produces:
      - grouped motif CSVs
      - raw motif CSVs
      - motif group mapping CSV
      - concept_to_motifs_full.csv
      - PDF plots (bars, polarity, tug-of-war, channel bars, topomaps)

    Outputs saved under: {root_dir}/{cfg.motif_out_subdir}/

    Returns dict of key output paths.
    """
    root = Path(root_dir or cfg.save_dir)
    if not root.exists():
        raise RuntimeError(f"Root dir not found: {root}")

    n_folds = int(cfg.motif_num_folds) if cfg.motif_num_folds is not None else int(cfg.n_folds)
    out_dir = root / cfg.motif_out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    concept_motif_csv = out_dir / "concept_to_motifs_full.csv"

    all_mdd: List[Dict[str, float]] = []
    all_healthy: List[Dict[str, float]] = []

    used_folds = 0
    for i in range(n_folds):
        fold_path = root / f"fold_{i}"
        concept_file = fold_path / "concept_labels.txt"
        explanation_file = fold_path / "subject_explanations.txt"

        if not concept_file.exists() or not explanation_file.exists():
            print(f"[WARNING] Missing fold files in: {fold_path}")
            continue

        used_folds += 1
        concept_to_motifs = parse_concepts(concept_file, keep_frac=float(cfg.motif_keep_frac))
        mdd, healthy = parse_subjects(explanation_file, concept_to_motifs)

        all_mdd.extend(mdd)
        all_healthy.extend(healthy)

    if used_folds == 0:
        raise RuntimeError("No folds had both concept_labels.txt and subject_explanations.txt.")

    df_mdd_all = summarize_motifs(all_mdd)
    df_healthy_all = summarize_motifs(all_healthy)

    df_mdd_grouped = summarize_grouped(df_mdd_all)
    df_healthy_grouped = summarize_grouped(df_healthy_all)

    # CSV artifacts
    (out_dir / "motifs_mdd_grouped.csv").write_text(df_mdd_grouped.to_csv(index=False), encoding="utf-8")
    (out_dir / "motifs_healthy_grouped.csv").write_text(df_healthy_grouped.to_csv(index=False), encoding="utf-8")
    (out_dir / "motifs_mdd_raw.csv").write_text(df_mdd_all.to_csv(index=False), encoding="utf-8")
    (out_dir / "motifs_healthy_raw.csv").write_text(df_healthy_all.to_csv(index=False), encoding="utf-8")

    motif_groups = pd.DataFrame({"Motif": pd.concat([df_mdd_all["Motif"], df_healthy_all["Motif"]], ignore_index=True)})
    motif_groups["Group"] = motif_groups["Motif"].apply(assign_group)
    motif_groups.drop_duplicates().to_csv(out_dir / "motif_group_mapping.csv", index=False)

    # concept->motif mappings across folds
    records = []
    for fold in range(n_folds):
        concept_file = root / f"fold_{fold}" / "concept_labels.txt"
        if not concept_file.exists():
            continue
        fold_concepts = parse_concepts(concept_file, keep_frac=float(cfg.motif_keep_frac))
        for concept_id, motifs in fold_concepts.items():
            for motif in motifs:
                records.append({"Fold": fold, "Concept": concept_id, "Motif": motif})

    pd.DataFrame(records).to_csv(concept_motif_csv, index=False)
    print(f"[INFO] Saved concept-to-motif mapping to: {concept_motif_csv}")

    # Plots (PDF only)
    plot_top_groups_bar(
        df_mdd_grouped,
        f"Top {cfg.motif_top_n_groups} MDD Motif Groups by Total Absolute Contribution",
        out_dir / "top_mdd_groups.pdf",
        top_n=int(cfg.motif_top_n_groups),
        )
    plot_top_groups_bar(
        df_healthy_grouped,
        f"Top {cfg.motif_top_n_groups} Healthy Motif Groups by Total Absolute Contribution",
        out_dir / "top_healthy_groups.pdf",
        top_n=int(cfg.motif_top_n_groups),
        )

    plot_polarity_scatter(
        df_mdd_grouped,
        "MDD Motif Group Polarity (Positive vs Negative Subject Count)",
        out_dir / "polarity_mdd.pdf",
        )
    plot_polarity_scatter(
        df_healthy_grouped,
        "Healthy Motif Group Polarity (Positive vs Negative Subject Count)",
        out_dir / "polarity_healthy.pdf",
        )

    plot_tug_of_war(
        df_mdd_grouped,
        df_healthy_grouped,
        out_dir / "group_contribs_tugofwar.pdf",
        top_k=int(cfg.motif_top_k_tugofwar),
        )

    channel_mdd = channel_importance(df_mdd_all)
    channel_healthy = channel_importance(df_healthy_all)

    plot_channel_contrib(channel_mdd, "Channel Contribution – MDD", out_dir / "channel_contrib_mdd.pdf")
    plot_channel_contrib(channel_healthy, "Channel Contribution – Healthy", out_dir / "channel_contrib_healthy.pdf")

    data_mdd = build_topomap_data(channel_mdd, ALL_CHANNELS)
    data_healthy = build_topomap_data(channel_healthy, ALL_CHANNELS)
    vmax = float(max(np.max(np.abs(data_mdd)), np.max(np.abs(data_healthy)))) or 1.0

    plot_topomap_contrib(data_mdd, "MDD – Channel Contribution", out_dir / "topomap_mdd_contrib.pdf", vmax=vmax, cmap="Reds")
    plot_topomap_contrib(data_healthy, "Healthy – Channel Contribution", out_dir / "topomap_healthy_contrib.pdf", vmax=vmax, cmap="Blues")

    print(f"[INFO] All artifacts saved under: {out_dir.resolve()}")

    return {
        "out_dir": str(out_dir),
        "concept_motif_csv": str(concept_motif_csv),
        "mdd_grouped_csv": str(out_dir / "motifs_mdd_grouped.csv"),
        "healthy_grouped_csv": str(out_dir / "motifs_healthy_grouped.csv"),
        "mdd_raw_csv": str(out_dir / "motifs_mdd_raw.csv"),
        "healthy_raw_csv": str(out_dir / "motifs_healthy_raw.csv"),
    }
