from pathlib import Path
import json
import importlib.util
import numpy as np
import pandas as pd
import torch

ROOT = Path(r"D:\data\lsy\vm_lsy_parent\lsy")
PRETRAIN = ROOT / "02_results" / "single_cell" / "20260520_SCP682_PPKO_release_v1" / "scripts" / "pretrain_scp682_ppko_attention_prior_v10.py"
GRAPH_DIR = ROOT / "01_data" / "pathway_prior" / "intermediate" / "global_phosphoprotein_heterograph_v10_measured_string700_top50"
TRAIN_DIR = ROOT / "01_data" / "single_cell" / "intermediate" / "phospho_perturb" / "decryptm_comparison_delta_v8"
PAN_RPPA = ROOT / "01_data" / "bulk_external" / "tcpa_rppa500_20260527" / "extracted" / "TCPA_TCGA_RPPA500.tsv"
PREV = ROOT / "02_results" / "clinical_validation" / "20260527_tcga_tcpa_drug_phospho_response_v2_pancan"
OUT = ROOT / "02_results" / "clinical_validation" / "20260531_tcga_tcpa_ppko_patient_response_v1_targeted_expanded"

MODELS = {
    "release_v10": ROOT / "02_results" / "single_cell" / "20260520_SCP682_PPKO_release_v1" / "models" / "SCP682_PPKO_best.pt",
    "v10b_300": ROOT / "02_results" / "single_cell" / "20260520_scp682_ppko_1_attention_prior_v10b_strong_contrast_300" / "models" / "scp682_ppko_attention_prior_v10_best.pt",
}

DRUG_TARGETS = {
    "Erlotinib": "EGFR",
    "Gefitinib": "EGFR",
    "Afatinib": "EGFR;ERBB2",
    "Lapatinib": "EGFR;ERBB2",
    "Trametinib": "MAP2K1;MAP2K2",
    "Selumetinib": "MAP2K1;MAP2K2",
    "Vemurafenib": "BRAF",
    "Dabrafenib": "BRAF",
    "BRAF inhibitor": "BRAF",
    "Sorafenib": "RAF1;BRAF;KDR;FLT3;KIT;PDGFRB",
    "Sunitinib": "KDR;FLT1;FLT3;KIT;PDGFRA;PDGFRB",
    "Pazopanib": "KDR;FLT1;FLT4;PDGFRA;PDGFRB;KIT",
    "Axitinib": "KDR;FLT1;FLT4",
    "Cabozantinib": "MET;KDR;AXL;RET",
    "AZD2171": "KDR;FLT1;FLT4;KIT",
    "Imatinib": "ABL1;KIT;PDGFRA;PDGFRB",
    "Dasatinib": "ABL1;SRC;LCK;YES1;KIT",
    "Nilotinib": "ABL1;KIT;PDGFRA;PDGFRB",
    "Everolimus": "MTOR",
    "Temsirolimus": "MTOR",
    "ridaforolimus": "MTOR",
    "Cetuximab": "EGFR",
    "Panitumumab": "EGFR",
    "Trastuzumab": "ERBB2",
    "Pertuzumab": "ERBB2",
    "Ramucirumab": "KDR",
    "Bevacizumab": "KDR;FLT1;FLT4",
}

DIRECT_DRUGS = set(DRUG_TARGETS)

DRUG_CLASS_OVERRIDE = {
    "Cetuximab": "EGFR",
    "Panitumumab": "EGFR",
    "Trastuzumab": "ERBB2",
    "Pertuzumab": "ERBB2",
    "Ramucirumab": "VEGFR",
    "Bevacizumab": "VEGFR",
}

DRUG_MODALITY_OVERRIDE = {
    "Cetuximab": "targeted_antibody",
    "Panitumumab": "targeted_antibody",
    "Trastuzumab": "targeted_antibody",
    "Pertuzumab": "targeted_antibody",
    "Ramucirumab": "targeted_antibody",
    "Bevacizumab": "ligand_blocking_antibody_mapped_to_receptor",
}

TCPA_MARKER_GENES = {
    "EGFRPY1068": "EGFR",
    "HER2PY1248": "ERBB2",
    "MAPKPT202Y204": "MAPK1;MAPK3",
    "MEK1PS217S221": "MAP2K1",
    "AKTPS473": "AKT1;AKT2;AKT3",
    "AKTPT308": "AKT1;AKT2;AKT3",
    "MTORPS2448": "MTOR",
    "S6PS235S236": "RPS6",
    "S6PS240S244": "RPS6",
    "P70S6KPT389": "RPS6KB1",
    "SRCPY416": "SRC",
    "SRCPY527": "SRC",
    "CABLPY412": "ABL1",
    "SHCPY317": "SHC1",
    "BRAFPS445": "BRAF",
    "P38PT180Y182": "MAPK14",
    "JNKPT183Y185": "MAPK8;MAPK9",
    "SHP2PY542": "PTPN11",
    "STAT3PY705": "STAT3",
    "PDK1PS241": "PDPK1",
    "PRAS40PT246": "AKT1S1",
    "GSK3ALPHABETAPS21S9": "GSK3A;GSK3B",
    "GSK3PS9": "GSK3B",
    "TUBERINPT1462": "TSC2",
    "YAPPS127": "YAP1",
    "YB1PS102": "YBX1",
    "EPHA2PS897": "EPHA2",
    "EPHA2PY588": "EPHA2",
    "FRS2ALPHAPY196": "FRS2",
    "RBPS807S811": "RB1",
    "CREBPS133": "CREB1",
    "CJUNPS73": "JUN",
    "HSP27PS82": "HSPB1",
    "EIF4EPS209": "EIF4E",
    "X4EBP1PS65": "EIF4EBP1",
    "X4EBP1PT37T46": "EIF4EBP1",
    "X4EBP1PT70": "EIF4EBP1",
}


def load_pretrain_module():
    spec = importlib.util.spec_from_file_location("ppko_v10", PRETRAIN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def auc_from_scores(y_true, scores):
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    mask = np.isfinite(y) & np.isfinite(s)
    y = y[mask]
    s = s[mask]
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    wins = 0.0
    for p in pos:
        wins += np.sum(p > neg)
        wins += 0.5 * np.sum(p == neg)
    return float(wins / (len(pos) * len(neg)))


def summarize(df, cols):
    rows = []
    for key, sub in df.groupby(cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        y = sub["response_binary"].to_numpy(float)
        rec = {c: v for c, v in zip(cols, key)}
        rec.update({
            "n": int(len(sub)),
            "n_responder": int(np.sum(y == 1)),
            "n_non_responder": int(np.sum(y == 0)),
        })
        for score_col in ["ppko_abs_delta_mean", "ppko_abs_delta_top200_mean", "ppko_target_prior_abs_mean", "ppko_observed_site_abs_mean"]:
            rec[f"{score_col}_auc"] = auc_from_scores(y, sub[score_col].to_numpy(float))
            rec[f"{score_col}_mean_responder"] = float(np.nanmean(sub.loc[sub["response_binary"].eq(1), score_col])) if np.any(y == 1) else np.nan
            rec[f"{score_col}_mean_non_responder"] = float(np.nanmean(sub.loc[sub["response_binary"].eq(0), score_col])) if np.any(y == 0) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def make_generic_baseline(n_sites):
    base = np.load(TRAIN_DIR / "arrays" / "baseline_matrix.npy").astype(np.float32)
    valid = np.load(TRAIN_DIR / "arrays" / "valid_mask.npy").astype(bool)
    out = np.zeros(n_sites, dtype=np.float32)
    for j in range(n_sites):
        vals = base[valid[:, j], j]
        out[j] = np.nanmedian(vals) if len(vals) else 0.0
    return out


def build_tcpa_projection(sites, rppa):
    gene_to_sites = {}
    for _, row in sites.iterrows():
        idx = int(row["target_index"])
        genes = str(row.get("site_gene_list", "")).replace(",", ";").split(";")
        for g in genes:
            g = g.strip().upper()
            if g and g != "NAN":
                gene_to_sites.setdefault(g, []).append(idx)

    marker_to_indices = {}
    for marker, genes in TCPA_MARKER_GENES.items():
        if marker not in rppa.columns:
            continue
        idxs = []
        for g in genes.split(";"):
            idxs.extend(gene_to_sites.get(g.strip().upper(), []))
        idxs = sorted(set(idxs))
        if idxs:
            marker_to_indices[marker] = idxs
    return marker_to_indices


def patient_baseline(sample_row, generic_base, marker_to_indices):
    base = generic_base.copy()
    mask = np.zeros(len(base), dtype=bool)
    for marker, idxs in marker_to_indices.items():
        value = sample_row.get(marker, np.nan)
        if not np.isfinite(value):
            continue
        for idx in idxs:
            base[idx] = float(value)
            mask[idx] = True
    return base.astype(np.float32), mask


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "reports").mkdir(exist_ok=True)

    module = load_pretrain_module()
    rppa = pd.read_csv(PAN_RPPA, sep="\t").drop_duplicates("sample_id").set_index("sample_id")
    raw_clinical = pd.read_csv(PREV / "pancan_tcpa_drug_pairs_with_signature_status.tsv", sep="\t")
    clinical = raw_clinical.copy()
    clinical = clinical[clinical["drug_name"].isin(DIRECT_DRUGS)].copy()
    clinical["target_genes"] = clinical["drug_name"].map(DRUG_TARGETS)
    clinical = clinical[clinical["target_genes"].notna()].copy().reset_index(drop=True)
    clinical["drug_class_original"] = clinical["drug_class"]
    clinical["drug_modality_original"] = clinical["drug_modality"]
    clinical["drug_class"] = clinical["drug_name"].map(DRUG_CLASS_OVERRIDE).fillna(clinical["drug_class"])
    clinical["drug_modality"] = clinical["drug_name"].map(DRUG_MODALITY_OVERRIDE).fillna(clinical["drug_modality"])

    drug_filter_audit = pd.DataFrame([
        {
            "stage": "tcpa_response_binary_all_drugs",
            "n_records": int(len(raw_clinical)),
            "n_patients": int(raw_clinical["patient12"].nunique()),
            "n_responder": int(raw_clinical["response_binary"].sum()),
            "n_non_responder": int((raw_clinical["response_binary"] == 0).sum()),
        },
        {
            "stage": "targeted_expanded_mappable_drugs",
            "n_records": int(len(clinical)),
            "n_patients": int(clinical["patient12"].nunique()),
            "n_responder": int(clinical["response_binary"].sum()),
            "n_non_responder": int((clinical["response_binary"] == 0).sum()),
        },
    ])
    drug_filter_audit.to_csv(OUT / "tables" / "targeted_expansion_filter_audit.tsv", sep="\t", index=False)
    (
        clinical.groupby(["drug_name", "target_genes", "drug_class", "drug_modality"], dropna=False)
        .agg(
            n_records=("patient12", "size"),
            n_patients=("patient12", "nunique"),
            n_responder=("response_binary", "sum"),
        )
        .reset_index()
        .assign(n_non_responder=lambda x: x["n_records"] - x["n_responder"])
        .sort_values(["n_records", "drug_name"], ascending=[False, True])
        .to_csv(OUT / "tables" / "targeted_expansion_drug_inclusion_audit.tsv", sep="\t", index=False)
    )

    outputs = []
    inventories = {}
    for model_name, model_path in MODELS.items():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        sites = pd.DataFrame(ckpt["sites"])
        if "site_gene_list" not in sites.columns:
            graph_sites = pd.read_csv(GRAPH_DIR / "tables" / "site_nodes.tsv", sep="\t")
            sites = sites.merge(graph_sites[["target_index", "site_gene_list"]], on="target_index", how="left")
        n_sites = len(sites)
        marker_to_indices = build_tcpa_projection(sites, rppa)
        generic_base = make_generic_baseline(n_sites)

        context_table = clinical[["target_genes"]].copy().reset_index(drop=True)
        context_table["action_type"] = "inhibition"
        _, protein_context, graph_prior, proteins = module.build_global_signed_inputs(context_table, GRAPH_DIR)

        model = module.AttentionPriorManifoldV10(
            n_sites,
            len(proteins),
            hidden=ckpt["args"]["hidden"],
            latent=ckpt["args"]["latent"],
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        rows = []
        with torch.no_grad():
            for i, (_, row) in enumerate(clinical.iterrows()):
                if row["sample_id"] not in rppa.index:
                    continue
                base, obs_mask = patient_baseline(rppa.loc[row["sample_id"]], generic_base, marker_to_indices)
                model_mask = obs_mask.copy()
                if model_mask.sum() == 0:
                    model_mask[:] = True
                b = torch.as_tensor(base[None, :], dtype=torch.float32, device=device)
                m = torch.as_tensor(model_mask[None, :], dtype=torch.bool, device=device)
                pc = torch.as_tensor(protein_context[i:i+1], dtype=torch.float32, device=device)
                gp = torch.as_tensor(graph_prior[i:i+1], dtype=torch.float32, device=device)
                pred, graph_delta, latent_delta, residual_delta, dz, attn = model(b, m, pc, gp)
                pred_np = pred.detach().cpu().numpy()[0]
                gp_np = graph_prior[i]
                obs_idx = np.where(obs_mask)[0]
                top_idx = np.argsort(-np.abs(pred_np))[:200]
                prior_idx = np.argsort(-np.abs(gp_np))[:200]
                rec = row.to_dict()
                rec["model_name"] = model_name
                rec["model_path"] = str(model_path)
                rec["n_observed_projected_sites"] = int(obs_mask.sum())
                rec["n_tcpa_markers_projected"] = int(sum(np.isfinite(rppa.loc[row["sample_id"], list(marker_to_indices.keys())].to_numpy(dtype=float))))
                rec["ppko_abs_delta_mean"] = float(np.nanmean(np.abs(pred_np)))
                rec["ppko_abs_delta_top200_mean"] = float(np.nanmean(np.abs(pred_np[top_idx])))
                rec["ppko_target_prior_abs_mean"] = float(np.nanmean(np.abs(pred_np[prior_idx])))
                rec["ppko_observed_site_abs_mean"] = float(np.nanmean(np.abs(pred_np[obs_idx]))) if len(obs_idx) else np.nan
                rec["ppko_signed_delta_mean"] = float(np.nanmean(pred_np))
                rows.append(rec)

        pred_df = pd.DataFrame(rows)
        pred_df.to_csv(OUT / "tables" / f"{model_name}_patient_predictions.tsv", sep="\t", index=False)
        outputs.append(pred_df)

        inventories[model_name] = {
            "model_path": str(model_path),
            "n_patient_drug_rows": int(len(pred_df)),
            "n_patients": int(pred_df["patient12"].nunique()) if len(pred_df) else 0,
            "n_observed_projected_sites_median": float(np.nanmedian(pred_df["n_observed_projected_sites"])) if len(pred_df) else np.nan,
            "n_markers_projected": int(len(marker_to_indices)),
            "marker_to_n_sites": {k: len(v) for k, v in marker_to_indices.items()},
        }

    all_pred = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
    all_pred.to_csv(OUT / "tables" / "all_model_patient_predictions.tsv", sep="\t", index=False)
    if len(all_pred):
        summarize(all_pred, ["model_name"]).to_csv(OUT / "tables" / "model_level_summary.tsv", sep="\t", index=False)
        summarize(all_pred, ["model_name", "drug_class"]).to_csv(OUT / "tables" / "model_drug_class_summary.tsv", sep="\t", index=False)
        summarize(all_pred, ["model_name", "drug_name"]).to_csv(OUT / "tables" / "model_drug_summary.tsv", sep="\t", index=False)
        (
            all_pred.groupby(["model_name", "Cancer"], dropna=False)
            .agg(
                n_records=("patient12", "size"),
                n_patients=("patient12", "nunique"),
                n_responder=("response_binary", "sum"),
            )
            .reset_index()
            .assign(n_non_responder=lambda x: x["n_records"] - x["n_responder"])
            .sort_values(["model_name", "n_responder", "n_records"], ascending=[True, False, False])
            .to_csv(OUT / "tables" / "model_cancer_type_response_summary.tsv", sep="\t", index=False)
        )
        (
            all_pred.groupby(["Cancer"], dropna=False)
            .agg(
                n_records=("patient12", "size"),
                n_patients=("patient12", "nunique"),
                n_responder=("response_binary", "sum"),
            )
            .reset_index()
            .assign(n_non_responder=lambda x: x["n_records"] - x["n_responder"])
            .sort_values(["n_responder", "n_records"], ascending=[False, False])
            .to_csv(OUT / "tables" / "cancer_type_response_summary_all_models.tsv", sep="\t", index=False)
        )
        primary = all_pred[all_pred["model_name"].eq("v10b_300")].copy()
        if len(primary):
            (
                primary.groupby(["Cancer"], dropna=False)
                .agg(
                    n_records=("patient12", "size"),
                    n_patients=("patient12", "nunique"),
                    n_responder=("response_binary", "sum"),
                )
                .reset_index()
                .assign(n_non_responder=lambda x: x["n_records"] - x["n_responder"])
                .sort_values(["n_responder", "n_records"], ascending=[False, False])
                .to_csv(OUT / "tables" / "v10b_300_cancer_type_response_summary.tsv", sep="\t", index=False)
            )

    (OUT / "reports" / "inventory.json").write_text(json.dumps(inventories, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = [
        "# TCGA-TCPA SCP682-PPKO 病人反应验证",
        "",
        "输入为 TCPA 全癌种病人基线 RPPA，条件为 Ding 表中分子靶向药对应的带符号靶点基因。",
        "TCPA 抗体不能精确映射到全部质谱肽段，所以这里只打开同基因外部观测位点作为基线掩码。",
        "",
        "输出标量不是手工通路平均分，而是 SCP682-PPKO 对每个病人-药物对预测出的扰动幅度。",
        "",
        "主要表：",
        "- tables/all_model_patient_predictions.tsv",
        "- tables/model_level_summary.tsv",
        "- tables/model_drug_class_summary.tsv",
        "- tables/model_drug_summary.tsv",
        "- tables/model_cancer_type_response_summary.tsv",
        "- tables/v10b_300_cancer_type_response_summary.tsv",
        "- tables/cancer_type_response_summary_all_models.tsv",
        "- tables/targeted_expansion_filter_audit.tsv",
        "- tables/targeted_expansion_drug_inclusion_audit.tsv",
        "- reports/inventory.json",
        "",
        "新增映射：Cetuximab/Panitumumab -> EGFR；Trastuzumab/Pertuzumab -> ERBB2；Ramucirumab -> KDR；Bevacizumab -> KDR;FLT1;FLT4。",
        "Rituximab、化疗、内分泌治疗、免疫治疗不纳入。",
    ]
    (OUT / "MANIFEST.md").write_text("\n".join(manifest), encoding="utf-8")

    print(OUT)
    print(json.dumps(inventories, ensure_ascii=False, indent=2))
    if len(all_pred):
        print(pd.read_csv(OUT / "tables" / "model_level_summary.tsv", sep="\t").to_string(index=False))


if __name__ == "__main__":
    main()
