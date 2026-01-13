import os
import json
import joblib
import numpy as np

from .explain import feature_family


def save_fold_artifacts(
        fold_dir,
        scaler,
        selector,
        feat_names,
        ch_names_ref,
        train_sid_maps,
        val_sid_maps,
        concept_labels=None,
        subj_summary_named=None,
        subj_probs_named=None,
        subj_conf_named=None,
):
    os.makedirs(fold_dir, exist_ok=True)

    if scaler is not None:
        joblib.dump(scaler, os.path.join(fold_dir, "scaler.joblib"))
    if selector is not None:
        joblib.dump(selector, os.path.join(fold_dir, "selector.joblib"))

    if feat_names is not None:
        np.save(
            os.path.join(fold_dir, "selected_features.npy"),
            np.array(feat_names, dtype=object),
            allow_pickle=True,
        )

    if ch_names_ref is not None:
        np.save(
            os.path.join(fold_dir, "channels.npy"),
            np.array(ch_names_ref, dtype=object),
            allow_pickle=True,
        )

    def _dump_sid_maps(name, sid_maps):
        if sid_maps is None:
            return
        path = os.path.join(fold_dir, f"{name}_sid_maps.json")
        out = {
            "sid_to_int": {str(k): int(v) for k, v in sid_maps["sid_to_int"].items()},
            "int_to_sid": {str(k): str(v) for k, v in sid_maps["int_to_sid"].items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

    _dump_sid_maps("train", train_sid_maps)
    _dump_sid_maps("val", val_sid_maps)

    if concept_labels is not None:
        np.save(
            os.path.join(fold_dir, "concept_labels.npy"),
            np.array(concept_labels, dtype=object),
            allow_pickle=True,
        )
        with open(os.path.join(fold_dir, "concept_labels.txt"), "w", encoding="utf-8") as f:
            for ci, rows in enumerate(concept_labels):
                f.write(f"Concept {ci}:\n")
                for (fname, sgn, rabs) in rows:
                    sign = "+" if sgn >= 0 else "-"
                    fam = feature_family(fname)
                    fam_txt = f" [{fam}]" if fam != "other" else ""
                    f.write(f"  {sign} {fname}{fam_txt}  |r|={rabs:.3f}\n")
                f.write("\n")

    if subj_summary_named is not None:
        arr = np.array(list(subj_summary_named.items()), dtype=object)
        np.save(os.path.join(fold_dir, "subject_summaries.npy"), arr, allow_pickle=True)

        with open(os.path.join(fold_dir, "subject_explanations.txt"), "w", encoding="utf-8") as f:
            for sid in sorted(subj_summary_named.keys()):
                s = subj_summary_named[sid]
                f.write(f"Subject {sid}\n")

                if subj_probs_named is not None and sid in subj_probs_named:
                    p = float(subj_probs_named[sid])
                    f.write(f"  p(MDD) = {p:.6g}\n")

                if subj_conf_named is not None and sid in subj_conf_named:
                    conf = float(subj_conf_named[sid])
                    f.write(f"  confidence = {conf:.3f}\n")

                f.write("  Top concepts (idx : contrib):\n")
                for (ci, v) in s["topk"]:
                    if concept_labels is not None and len(concept_labels[ci]) > 0:
                        top_label = concept_labels[ci][0][0]
                        fam = feature_family(top_label)
                        fam_txt = f" [{fam}]" if fam else ""
                    else:
                        top_label, fam_txt = "n/a", ""
                    stab = s["stability"].get(ci, 0.0)
                    f.write(f"    C{ci}: {v:+.4f}  ← {top_label}{fam_txt}  (stability={stab:.2f})\n")
                f.write("\n")
