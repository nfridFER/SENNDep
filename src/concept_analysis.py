import os
import re
import csv
from collections import defaultdict
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LINE_RE = re.compile(r"^[\+\-]\s+(\S+)\s+\[([^\]]+)\]\s+\|r\|=([0-9.]+)")
TOKEN_RE = re.compile(r"(?P<feat_name>.+?)\[(?P<ch>[A-Za-z0-9\-]+)\](?:__(?P<stat>\w+))?")

REGION_MAP = {
    "Fp1": "frontal", "Fp2": "frontal", "F3": "frontal", "F4": "frontal", "Fz": "frontal",
    "C3": "central", "C4": "central", "Cz": "central",
    "P3": "parietal", "P4": "parietal", "Pz": "parietal",
    "T3": "temporal", "T4": "temporal", "T5": "temporal", "T6": "temporal",
    "O1": "occipital", "O2": "occipital",
}


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "patch.edgecolor": "none",
            "patch.force_edgecolor": False,
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "grid.linestyle": "none",
            "grid.alpha": 0.4,
        }
    )


def _split_channels(ch: str) -> List[str]:
    return ch.split("-") if "-" in ch else [ch]


def _parse_feature(token: str, group: str) -> Optional[Tuple[str, str, str, str, Optional[str], str]]:
    m = TOKEN_RE.match(token)
    if not m:
        return None

    feat_name = m.group("feat_name")
    channel = m.group("ch")
    stat = m.group("stat") or "value"

    base_type = feat_name
    band = None

    if feat_name.startswith("relpow_"):
        base_type = "relpow"
        band = feat_name.split("_", 1)[1]
    elif feat_name.startswith("coh_"):
        base_type = "coh"
        band = feat_name.split("_", 1)[1]
    elif feat_name.startswith("coherence_"):
        base_type = "coh"
        band = feat_name.split("_", 1)[1]

    return feat_name, channel, stat, base_type, band, group


def _parse_concept_labels_txt(
        path: str,
        fold_id: int,
        feature_stats: Dict[Tuple, Dict],
        channel_stats: Dict[str, Dict],
        band_stats: Dict[str, Dict],
) -> None:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            m = LINE_RE.match(line)
            if not m:
                continue

            token = m.group(1)
            group = m.group(2)
            r_abs = float(m.group(3))

            parsed = _parse_feature(token, group)
            if parsed is None:
                continue

            feat_name, ch, stat, _, band, group = parsed

            feat_key = (feat_name, ch, stat, band, group)
            fs = feature_stats[feat_key]
            fs.setdefault("count", 0)
            fs.setdefault("folds", set())
            fs.setdefault("sum_r", 0.0)
            fs.setdefault("max_r", 0.0)
            fs["count"] += 1
            fs["folds"].add(fold_id)
            fs["sum_r"] += r_abs
            fs["max_r"] = max(fs["max_r"], r_abs)

            for ch_single in _split_channels(ch):
                cs = channel_stats[ch_single]
                cs.setdefault("count", 0)
                cs.setdefault("folds", set())
                cs.setdefault("sum_r", 0.0)
                cs.setdefault("max_r", 0.0)
                cs["count"] += 1
                cs["folds"].add(fold_id)
                cs["sum_r"] += r_abs
                cs["max_r"] = max(cs["max_r"], r_abs)

            if band is not None:
                bs = band_stats[band]
                bs.setdefault("count", 0)
                bs.setdefault("folds", set())
                bs.setdefault("sum_r", 0.0)
                bs.setdefault("max_r", 0.0)
                bs["count"] += 1
                bs["folds"].add(fold_id)
                bs["sum_r"] += r_abs
                bs["max_r"] = max(bs["max_r"], r_abs)


def _finalize_stats(d: Dict) -> None:
    for _, v in d.items():
        v["fold_count"] = len(v["folds"])
        v["mean_r"] = v["sum_r"] / v["count"] if v["count"] > 0 else 0.0


def _write_csv_summaries(
        feature_stats: Dict[Tuple, Dict],
        channel_stats: Dict[str, Dict],
        band_stats: Dict[str, Dict],
        out_dir: str,
        out_basename: str,
) -> Tuple[str, str, str]:
    _finalize_stats(feature_stats)
    _finalize_stats(channel_stats)
    _finalize_stats(band_stats)

    feat_csv = os.path.join(out_dir, f"{out_basename}_features_summary.csv")
    ch_csv = os.path.join(out_dir, f"{out_basename}_channels_summary.csv")
    band_csv = os.path.join(out_dir, f"{out_basename}_bands_summary.csv")

    with open(feat_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["feat_name", "channel", "stat", "band", "group",
             "count", "fold_count", "sum_r", "mean_r", "max_r", "folds_str"]
        )
        for (feat_name, ch, stat, band, group), s in feature_stats.items():
            folds_str = ",".join(str(x) for x in sorted(s["folds"]))
            w.writerow(
                [
                    feat_name,
                    ch,
                    stat,
                    band if band is not None else "",
                    group,
                    s["count"],
                    s["fold_count"],
                    f"{s['sum_r']:.6f}",
                    f"{s['mean_r']:.6f}",
                    f"{s['max_r']:.6f}",
                    folds_str,
                ]
            )

    with open(ch_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["channel", "count", "fold_count", "sum_r", "mean_r", "max_r"])
        for ch, s in channel_stats.items():
            w.writerow([ch, s["count"], s["fold_count"], f"{s['sum_r']:.6f}", f"{s['mean_r']:.6f}", f"{s['max_r']:.6f}"])

    with open(band_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["band", "count", "fold_count", "sum_r", "mean_r", "max_r"])
        for band, s in band_stats.items():
            w.writerow([band, s["count"], s["fold_count"], f"{s['sum_r']:.6f}", f"{s['mean_r']:.6f}", f"{s['max_r']:.6f}"])

    return feat_csv, ch_csv, band_csv


def _save_top_channels_plot(ch_csv: str, out_dir: str, top_k: int = 15) -> str:
    df_ch = pd.read_csv(ch_csv).sort_values("mean_r", ascending=False).head(top_k)

    plt.figure(figsize=(6, 5))
    plt.barh(df_ch["channel"], df_ch["mean_r"])
    plt.gca().invert_yaxis()
    plt.xlabel("mean |r| across features/folds")
    plt.tight_layout()

    out_path = os.path.join(out_dir, "top_channels.pdf")
    plt.savefig(out_path, format="pdf")
    plt.close()
    return out_path


def _save_band_relevance_plot(band_csv: str, out_dir: str) -> str:
    df_b = pd.read_csv(band_csv).sort_values("mean_r", ascending=False)

    plt.figure(figsize=(5, 4))
    plt.bar(df_b["band"], df_b["mean_r"])
    plt.ylabel("mean |r|")
    plt.xlabel("Band")
    plt.tight_layout()

    out_path = os.path.join(out_dir, "band_relevance.pdf")
    plt.savefig(out_path, format="pdf")
    plt.close()
    return out_path


def _save_cross_fold_presence_heatmap(
        feat_csv: str,
        out_dir: str,
        top_k: int = 30,
        min_fold_count: int = 1,
) -> Optional[str]:
    df = pd.read_csv(feat_csv)
    df = df[df["fold_count"] >= min_fold_count].copy()
    if df.empty:
        return None

    df["label"] = df.apply(lambda r: f"{r['feat_name']}[{r['channel']}] ({r['group']})", axis=1)
    df = df.sort_values(["mean_r", "fold_count"], ascending=False).head(top_k)

    def parse_folds(s):
        if isinstance(s, str) and s.strip():
            return [int(x) for x in s.split(",") if x.strip() != ""]
        return []

    df["fold_list"] = df["folds_str"].apply(parse_folds)

    n_folds = 0
    for fl in df["fold_list"]:
        if fl:
            n_folds = max(n_folds, max(fl) + 1)
    if n_folds == 0:
        return None

    mat = np.zeros((len(df), n_folds), dtype=int)
    for i, fl in enumerate(df["fold_list"].tolist()):
        for f in fl:
            mat[i, f] = 1

    plt.figure(figsize=(7, max(4, 0.35 * len(df))))
    plt.imshow(mat, aspect="auto", cmap="Greys")
    plt.colorbar(label="present in fold")
    plt.yticks(range(len(df)), df["label"])
    plt.xticks(range(n_folds), [f"fold {i}" for i in range(n_folds)], rotation=45)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "cross_fold_presence_heatmap.pdf")
    plt.savefig(out_path, format="pdf")
    plt.close()
    return out_path


def _save_region_band_heatmap(feat_csv: str, out_dir: str) -> Optional[str]:
    df_f = pd.read_csv(feat_csv)

    def get_region(channel: str) -> str:
        return REGION_MAP.get(channel, "other")

    df_f["region"] = df_f["channel"].astype(str).apply(get_region)

    df_rel = df_f[df_f["feat_name"].astype(str).str.startswith("relpow_")].copy()
    if df_rel.empty:
        return None

    pivot = df_rel.pivot_table(index="region", columns="band", values="mean_r", aggfunc="mean")
    if pivot.empty:
        return None

    plt.figure(figsize=(6, 4))
    plt.imshow(pivot.values, aspect="auto", cmap="viridis")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.colorbar(label="mean |r|")
    plt.tight_layout()

    out_path = os.path.join(out_dir, "region_band_relevance.pdf")
    plt.savefig(out_path, format="pdf")
    plt.close()
    return out_path


def _save_feature_group_relevance(feat_csv: str, out_dir: str) -> Optional[str]:
    df_f = pd.read_csv(feat_csv)
    if df_f.empty or "group" not in df_f.columns:
        return None

    df_group = (
        df_f.groupby("group", dropna=False)
        .agg(mean_r=("mean_r", "mean"), count=("feat_name", "count"))
        .sort_values("mean_r", ascending=False)
    )

    plt.figure(figsize=(6, 4))
    plt.bar(df_group.index.astype(str), df_group["mean_r"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("mean |r|")
    plt.tight_layout()

    out_path = os.path.join(out_dir, "feature_group_relevance.pdf")
    plt.savefig(out_path, format="pdf")
    plt.close()
    return out_path


def run_concept_analysis(cfg, root_dir: Optional[str] = None) -> Dict[str, object]:
    """
    Parse fold_*/concept_labels.txt and write summaries + plots into:
      {root_dir}/{cfg.concept_out_subdir}/

    Returns dict with output paths.
    """
    _set_plot_style()

    root_dir = root_dir or cfg.save_dir
    if not os.path.isdir(root_dir):
        raise RuntimeError(f"Root dir not found: {root_dir}")

    out_dir = os.path.join(root_dir, cfg.concept_out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    # find fold dirs
    fold_dirs = []
    for name in os.listdir(root_dir):
        full = os.path.join(root_dir, name)
        if os.path.isdir(full) and name.startswith("fold"):
            fold_dirs.append((name, full))

    def fold_key(item):
        name, _ = item
        digits = "".join(ch for ch in name if ch.isdigit())
        return int(digits) if digits else 9999

    fold_dirs = sorted(fold_dirs, key=fold_key)
    if not fold_dirs:
        raise RuntimeError(f"No fold* subdirectories found in {root_dir}")

    feature_stats = defaultdict(dict)
    channel_stats = defaultdict(dict)
    band_stats = defaultdict(dict)

    used_folds = 0
    for fold_id, (name, path) in enumerate(fold_dirs):
        concept_path = os.path.join(path, "concept_labels.txt")
        if not os.path.isfile(concept_path):
            print(f"[WARN] no concept_labels.txt in {path}, skipping")
            continue
        used_folds += 1
        print(f"Parsing fold {fold_id} ({name}) from {concept_path}")
        _parse_concept_labels_txt(concept_path, fold_id, feature_stats, channel_stats, band_stats)

    if used_folds == 0:
        raise RuntimeError("No folds with concept_labels.txt found.")

    feat_csv, ch_csv, band_csv = _write_csv_summaries(
        feature_stats=feature_stats,
        channel_stats=channel_stats,
        band_stats=band_stats,
        out_dir=out_dir,
        out_basename=cfg.concept_out_basename,
    )

    figs = []
    figs.append(_save_top_channels_plot(ch_csv, out_dir, top_k=cfg.concept_top_k_channels))
    figs.append(_save_band_relevance_plot(band_csv, out_dir))

    heatmap = _save_cross_fold_presence_heatmap(
        feat_csv,
        out_dir,
        top_k=cfg.concept_top_k_presence,
        min_fold_count=cfg.concept_min_fold_count,
    )
    if heatmap:
        figs.append(heatmap)

    rb = _save_region_band_heatmap(feat_csv, out_dir)
    if rb:
        figs.append(rb)

    fg = _save_feature_group_relevance(feat_csv, out_dir)
    if fg:
        figs.append(fg)

    print("All interpretability artifacts in:", os.path.abspath(out_dir))

    return {
        "out_dir": out_dir,
        "feat_csv": feat_csv,
        "ch_csv": ch_csv,
        "band_csv": band_csv,
        "figs": figs,
    }
