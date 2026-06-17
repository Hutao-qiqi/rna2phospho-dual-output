#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROBUST_CANDIDATES = ["ridge_direct", "parent_only", "rna_direct"]
CVAE_CANDIDATES = [
    "v3_train_oof",
    "v3_1_1_feedback_prior",
    "v3_1_2_target_attention",
    "v3_6_ranking_coverage_loss",
]

DEFAULT_CONTEXT_TO_STUDY = {
    "BRCA_TCGA": "PDC000174",
    "COAD_PROSPECTIVE": "PDC000117",
    "GBM_DISCOVERY": "PDC000205",
    "HNSCC": "PDC000222",
    "CCRCC": "PDC000128",
    "NON_CCRCC": "PDC000465",
    "LUAD": "PDC000149",
    "LSCC": "PDC000232",
    "OV_TCGA": "PDC000115",
    "PDA": "PDC000271",
    "STAD": "PDC000615",
    "UCEC": "PDC000126",
}


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class SCP682V4Engine:
    def __init__(
        self,
        package_dir: Path,
        device: str = "auto",
        batch_size: int = 32,
        bank_k: int = 8,
        bank_chunk: int = 512,
        ridge_alpha: float = 10.0,
        seed: int = 20260508,
    ) -> None:
        self.package_dir = Path(package_dir).resolve()
        self.engine_dir = self.package_dir / "v4_engine"
        self.code_dir = self.engine_dir / "code"
        self.data_dir = self.engine_dir / "data" / "pancancer_multi_task_locked_v2"
        self.args = argparse.Namespace(
            batch_size=int(batch_size),
            bank_k=int(bank_k),
            bank_chunk=int(bank_chunk),
            ridge_alpha=float(ridge_alpha),
            seed=int(seed),
        )
        if device == "auto":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device if (str(device).startswith("cuda") and torch.cuda.is_available()) else "cpu")
        self._base = None
        self._public = None
        self._reference = None

    def _patch_total_module(self, module):
        module.ROOT = self.engine_dir
        module.DATA_DIR = self.engine_dir / "data" / "pancancer_multi_task_locked_v1"
        module.OUT = self.engine_dir / "unused_total_out"
        return module

    def _patch_model_module(self, module):
        module.ROOT = self.engine_dir
        module.DATA = self.data_dir
        module.OUT = self.engine_dir / "unused_model_out"
        module.TOTAL_SCRIPT = self.code_dir / "train_cptac_total_proteome_film_vae_z_direct_residual_20260428.py"
        module.KINASE_PRIOR = self.engine_dir / "data" / "copheeksa_model_phosphosite_kinase_predictions.tsv"

        def import_total_module_patched():
            total = import_module(module.TOTAL_SCRIPT, f"{module.__name__}_total_module")
            return self._patch_total_module(total)

        module.import_total_module = import_total_module_patched
        return module

    def _load_base(self):
        if self._base is not None:
            return self._base
        base = import_module(
            self.code_dir / "deploy_v38_luad_cas_and_evaluate_true_phosphosite_20260502.py",
            "scp682_portable_v4_base",
        )
        base.ROOT = self.engine_dir
        base.DATA = self.data_dir
        base.V2_SCRIPT = self.code_dir / "train_cptac_joint_total_phosphosite_residual_nas_v2_20260429.py"
        base.V3_SCRIPT = self.code_dir / "train_cptac_parent_residual_kinase_cvae_v3_20260502.py"
        base.V311_SCRIPT = self.code_dir / "train_cptac_parent_residual_kinase_cvae_v3_1_1_feedback_prior_20260502.py"
        base.V312_SCRIPT = self.code_dir / "train_cptac_parent_residual_kinase_cvae_v3_1_2_target_attention_20260502.py"
        base.V36_SCRIPT = self.code_dir / "train_cptac_parent_residual_kinase_cvae_v3_6_ranking_coverage_loss_20260502.py"
        base.V3_MODEL_DIR = self.engine_dir / "models" / "v3"
        base.V311_MODEL_DIR = self.engine_dir / "models" / "v3_1_1_feedback_prior"
        base.V312_MODEL_DIR = self.engine_dir / "models" / "v3_1_2_target_attention"
        base.V36_MODEL_DIR = self.engine_dir / "models" / "v3_6_ranking_coverage_loss"
        base.V3_OOF = self.engine_dir / "predictions" / "v3_train_oof_phosphosite_predictions.parquet"

        def import_module_patched(path: Path, name: str):
            mod = import_module(Path(path), name)
            if Path(path).name.startswith("train_cptac_total_proteome"):
                return self._patch_total_module(mod)
            if Path(path).name.startswith("train_cptac_parent_residual_kinase_cvae"):
                return self._patch_model_module(mod)
            return mod

        base.import_module = import_module_patched
        self._base = base
        return base

    def _load_public(self):
        if self._public is not None:
            return self._public
        public = import_module(
            self.code_dir / "predict_scp682_v4_0_public_bulk_20260508.py",
            "scp682_portable_v4_public",
        )
        self._public = public
        return public

    def load_reference(self) -> dict:
        if self._reference is not None:
            return self._reference
        ref_dir = self.package_dir / "reference"
        centroids = pd.read_parquet(ref_dir / "rna_context_centroids.parquet")
        context = pd.read_csv(ref_dir / "context_study_map.tsv", sep="\t")
        self._reference = {
            "centroids": centroids,
            "context": context,
            "context_to_study": dict(zip(context["cptac_cancer_label"].astype(str), context["cptac_study_id"].astype(str))),
        }
        return self._reference

    def infer_context(self, rna: pd.DataFrame) -> pd.DataFrame:
        ref = self.load_reference()
        centroids = ref["centroids"]
        genes = [g for g in centroids.columns if g in rna.columns]
        if len(genes) < 100:
            label = "LUAD"
            study = DEFAULT_CONTEXT_TO_STUDY[label]
            return pd.DataFrame({"sample_id": rna.index.astype(str), "cptac_cancer_label": label, "cptac_study_id": study})
        x = rna[genes].apply(pd.to_numeric, errors="coerce")
        c = centroids[genes].apply(pd.to_numeric, errors="coerce")
        x = x.fillna(x.median(axis=0)).fillna(0.0).to_numpy(dtype=np.float32)
        c = c.fillna(0.0).to_numpy(dtype=np.float32)
        x = x - x.mean(axis=1, keepdims=True)
        c = c - c.mean(axis=1, keepdims=True)
        x = x / np.sqrt((x * x).sum(axis=1, keepdims=True)).clip(min=1e-6)
        c = c / np.sqrt((c * c).sum(axis=1, keepdims=True)).clip(min=1e-6)
        best = np.argmax(x @ c.T, axis=1)
        labels = centroids.index.astype(str).to_numpy()[best]
        context_to_study = ref["context_to_study"]
        studies = [context_to_study.get(label, DEFAULT_CONTEXT_TO_STUDY.get(label, "PDC000149")) for label in labels]
        return pd.DataFrame({"sample_id": rna.index.astype(str), "cptac_cancer_label": labels, "cptac_study_id": studies})

    @staticmethod
    def normalize_manifest(rna: pd.DataFrame, manifest: pd.DataFrame | None, fallback: pd.DataFrame) -> pd.DataFrame:
        if manifest is None:
            manifest = fallback.copy()
        else:
            manifest = manifest.copy()
            if "sample_id" not in manifest.columns:
                manifest = manifest.reset_index().rename(columns={manifest.index.name or "index": "sample_id"})
            manifest["sample_id"] = manifest["sample_id"].astype(str)
            manifest = manifest.drop_duplicates("sample_id", keep="first")
            manifest = manifest.set_index("sample_id").reindex(rna.index.astype(str)).reset_index()
            auto = fallback.set_index("sample_id")
            for col in ["cptac_cancer_label", "cptac_study_id"]:
                if col not in manifest.columns:
                    manifest[col] = np.nan
                manifest[col] = manifest[col].where(manifest[col].notna(), auto.loc[manifest["sample_id"], col].to_numpy())
        manifest["sample_id"] = manifest["sample_id"].astype(str)
        manifest["cptac_cancer_label"] = manifest["cptac_cancer_label"].astype(str)
        manifest["cptac_study_id"] = manifest["cptac_study_id"].astype(str)
        return manifest

    @staticmethod
    def sample_center(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        med = df.median(axis=1, skipna=True)
        centered = df.sub(med, axis=0).astype(np.float32)
        offsets = pd.DataFrame({"sample_id": df.index.astype(str), "raw_prediction_sample_median": med.to_numpy(dtype=float)})
        return centered, offsets

    def predict(self, rna: pd.DataFrame, manifest: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
        base = self._load_base()
        public = self._load_public()
        base.seed_all(self.args.seed)

        rna = rna.copy()
        rna.index = rna.index.astype(str)
        auto_manifest = self.infer_context(rna)
        manifest = self.normalize_manifest(rna, manifest, auto_manifest)
        manifest_indexed = manifest.set_index("sample_id").loc[rna.index.astype(str)].reset_index()
        context_info = manifest_indexed.set_index("sample_id")[["cptac_cancer_label", "cptac_study_id"]].to_dict("index")

        def make_context_arrays(index: pd.Index, ckpt: dict) -> tuple[np.ndarray, np.ndarray]:
            cancer_map = {c: i for i, c in enumerate(ckpt["cancer_levels"])}
            study_map = {s: i for i, s in enumerate(ckpt["study_levels"])}
            cancer = []
            study = []
            for sid in index.astype(str):
                row = context_info[sid]
                c = row["cptac_cancer_label"]
                s = row["cptac_study_id"]
                if c not in cancer_map:
                    c = "LUAD"
                if s not in study_map:
                    s = DEFAULT_CONTEXT_TO_STUDY.get(c, "PDC000149")
                cancer.append(cancer_map[c])
                study.append(study_map[s])
            return np.asarray(cancer, dtype=np.int64), np.asarray(study, dtype=np.int64)

        base.make_context_arrays = make_context_arrays
        data = base.load_base_data(self.device)
        x_ext = base.build_external_feature_frame(rna, self.device)
        manifest_indexed = manifest_indexed.set_index("sample_id").loc[x_ext.index.astype(str)].reset_index()

        candidates: dict[str, pd.DataFrame] = {}
        v3_pred, v3_total = base.predict_v3_family(
            "v3_train_oof",
            base.V3_SCRIPT,
            base.V3_MODEL_DIR,
            "20260502_cptac_parent_residual_kinase_cvae_experimental_v3",
            data,
            x_ext,
            self.device,
            self.args,
        )
        candidates["v3_train_oof"] = v3_pred
        for label, script, model_dir, prefix in [
            ("v3_1_1_feedback_prior", base.V311_SCRIPT, base.V311_MODEL_DIR, "20260502_cptac_parent_residual_kinase_cvae_experimental_v3_1_1_feedback_prior"),
            ("v3_1_2_target_attention", base.V312_SCRIPT, base.V312_MODEL_DIR, "20260502_cptac_parent_residual_kinase_cvae_experimental_v3_1_2_target_attention"),
            ("v3_6_ranking_coverage_loss", base.V36_SCRIPT, base.V36_MODEL_DIR, "20260502_cptac_parent_residual_kinase_cvae_experimental_v3_6_ranking_coverage_loss"),
        ]:
            pred, _ = base.predict_v3_family(label, script, model_dir, prefix, data, x_ext, self.device, self.args, v3_external_for_feedback=v3_pred)
            candidates[label] = pred

        light = public.export_light_candidates_dynamic(base, data, x_ext, v3_total, manifest_indexed, self.args)
        candidates.update(light)
        common = sorted(set.intersection(*[set(candidates[n].columns) for n in ROBUST_CANDIDATES + CVAE_CANDIDATES]))
        robust = sum(candidates[n].reindex(index=x_ext.index, columns=common) for n in ROBUST_CANDIDATES) / float(len(ROBUST_CANDIDATES))
        cvae = sum(candidates[n].reindex(index=x_ext.index, columns=common) for n in CVAE_CANDIDATES) / float(len(CVAE_CANDIDATES))
        raw = (0.8 * robust + 0.2 * cvae).astype(np.float32)
        raw.index.name = "sample_id"
        centered, offsets = self.sample_center(raw)
        return {
            "v4_raw": raw,
            "v4_centered": centered,
            "sample_median_offsets": offsets,
            "manifest": manifest_indexed,
        }
