#!/usr/bin/env python3
"""在M2.2固定输入表示和位点基底上训练C、D、E三种近零降权版本。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.model_selection import KFold


SEED = 20260830
TARGET_TAU_MEDIAN = 0.344
SELECTIVE = np.asarray([15, 18, 23, 30, 41, 48, 55, 58], dtype=np.int64)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--variant", choices=("C", "D", "E", "F"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--site-corr-weight", type=float, default=0.20)
    parser.add_argument("--patient-corr-weight", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def transform_inputs(rna: np.ndarray, protein: np.ndarray, package: np.lib.npyio.NpzFile) -> np.ndarray:
    rna_z = np.nan_to_num((rna - package["rna_mean"]) / package["rna_scale"], nan=0.0).astype(np.float32)
    protein_z = np.nan_to_num((protein - package["protein_mean"]) / package["protein_scale"], nan=0.0).astype(np.float32)
    feature = np.concatenate(
        [rna_z @ package["rna_components"].T, protein_z @ package["protein_components"].T], axis=1
    )
    return ((feature - package["feature_mean"]) / package["feature_scale"]).astype(np.float32)


def fit_fixed_offset(
    truth: np.ndarray,
    protein: np.ndarray,
    studies: np.ndarray,
    training: np.ndarray,
    parent_index: np.ndarray,
) -> np.ndarray:
    global_site = np.nan_to_num(np.nanmean(truth[training], axis=0), nan=0.0).astype(np.float32)
    global_count = np.isfinite(truth[training]).sum(axis=0).astype(np.float32)
    global_protein = np.nan_to_num(np.nanmean(protein[training], axis=0), nan=0.0).astype(np.float32)
    site_center = np.empty_like(truth, dtype=np.float32)
    protein_center = np.empty_like(protein, dtype=np.float32)
    training_mask = np.zeros(len(truth), dtype=bool)
    training_mask[training] = True
    for study in np.unique(studies):
        rows = np.flatnonzero(studies == study)
        fit_rows = rows[training_mask[rows]]
        if len(fit_rows):
            values = truth[fit_rows]
            count = np.isfinite(values).sum(axis=0).astype(np.float32)
            mean = np.divide(
                np.nansum(values, axis=0, dtype=np.float64), count,
                out=np.zeros(truth.shape[1], dtype=np.float64), where=count > 0,
            ).astype(np.float32)
            common = (count >= 5) & (global_count >= 5)
            offset = float(np.median(mean[common] - global_site[common])) if common.any() else 0.0
            fallback = global_site + offset
            mean = np.where(count > 0, mean, fallback)
            weight = count / (count + 2.0)
            site_center[rows] = (weight * mean + (1.0 - weight) * fallback)[None, :]
            local_protein = np.nanmean(protein[fit_rows], axis=0).astype(np.float32)
            local_protein = np.where(np.isfinite(local_protein), local_protein, global_protein)
            protein_center[rows] = local_protein[None, :]
        else:
            site_center[rows] = global_site[None, :]
            protein_center[rows] = global_protein[None, :]
    protein_centered = protein - protein_center
    residual = truth - site_center
    beta = np.zeros(truth.shape[1], dtype=np.float32)
    for start in range(0, truth.shape[1], 1000):
        stop = min(start + 1000, truth.shape[1])
        x = protein_centered[:, parent_index[start:stop]]
        y = residual[:, start:stop]
        observed = np.isfinite(x[training]) & np.isfinite(y[training])
        count = observed.sum(axis=0)
        numerator = np.where(observed, x[training] * y[training], 0.0).sum(axis=0)
        denominator = np.where(observed, x[training] ** 2, 0.0).sum(axis=0)
        slope = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-8)
        slope *= count / (count + 20.0)
        slope[count < 16] = 0.0
        beta[start:stop] = slope.astype(np.float32)
    parent = np.nan_to_num(protein_centered[:, parent_index], nan=0.0) * beta[None, :]
    return (site_center + parent).astype(np.float32)


def empirical_tau(residual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmedian(residual, axis=0).astype(np.float32)
    mad = (1.4826 * np.nanmedian(np.abs(residual - center[None, :]), axis=0)).astype(np.float32)
    tau = 0.5 * mad
    finite = tau[np.isfinite(tau) & (tau > 1e-6)]
    scale = TARGET_TAU_MEDIAN / float(np.median(finite)) if finite.size else 1.0
    tau = np.where(np.isfinite(tau) & (tau > 1e-6), tau * scale, TARGET_TAU_MEDIAN).astype(np.float32)
    center = np.nan_to_num(center, nan=0.0).astype(np.float32)
    return center, tau


class ValueModel(torch.nn.Module):
    def __init__(self, package: np.lib.npyio.NpzFile):
        super().__init__()
        self.linear = torch.nn.Linear(384, 128)
        with torch.no_grad():
            self.linear.weight.copy_(torch.from_numpy(package["ridge_coef"]))
            self.linear.bias.copy_(torch.from_numpy(package["ridge_intercept"]))

    def forward(self, feature: torch.Tensor, basis: torch.Tensor, site_mean: torch.Tensor) -> torch.Tensor:
        return self.linear(feature) @ basis + site_mean


class HurdleModel(torch.nn.Module):
    def __init__(self, package: np.lib.npyio.NpzFile, active_rate: np.ndarray):
        super().__init__()
        self.value = ValueModel(package)
        self.response = torch.nn.Linear(384, 128)
        torch.nn.init.zeros_(self.response.weight)
        torch.nn.init.zeros_(self.response.bias)
        clipped = np.clip(active_rate, 1e-4, 1 - 1e-4)
        self.site_bias = torch.nn.Parameter(torch.from_numpy(np.log(clipped / (1.0 - clipped)).astype(np.float32)))

    def forward(
        self,
        feature: torch.Tensor,
        basis: torch.Tensor,
        response_basis: torch.Tensor,
        site_mean: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.value(feature, basis, site_mean)
        logits = self.response(feature) @ response_basis + self.site_bias
        return value, logits


def train_model(
    variant: str,
    feature: np.ndarray,
    residual: np.ndarray,
    package: np.lib.npyio.NpzFile,
    basis_np: np.ndarray,
    site_mean_np: np.ndarray,
    updates: int,
    batch_size: int,
    learning_rate: float,
    site_corr_weight: float,
    patient_corr_weight: float,
    device: torch.device,
    seed: int,
) -> tuple[torch.nn.Module, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    center_np, tau_np = empirical_tau(residual)
    distance = np.abs(residual - center_np[None, :])
    observed_np = np.isfinite(residual)
    neutral_np = observed_np & (distance <= tau_np[None, :])
    active_np = observed_np & (distance >= 2.0 * tau_np[None, :])
    active_rate = np.divide(active_np.sum(0), observed_np.sum(0), out=np.zeros(residual.shape[1]), where=observed_np.sum(0) > 0)
    basis = torch.from_numpy(basis_np).to(device)
    response_basis = torch.from_numpy(np.abs(basis_np)).to(device)
    site_mean = torch.from_numpy(site_mean_np).to(device)
    tau = torch.from_numpy(tau_np).to(device)
    center = torch.from_numpy(center_np).to(device)
    model: torch.nn.Module
    if variant == "E":
        model = HurdleModel(package, active_rate)
    else:
        model = ValueModel(package)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    feature_tensor = torch.from_numpy(feature)
    residual_tensor = torch.from_numpy(np.nan_to_num(residual, nan=0.0).astype(np.float32))
    observed_tensor = torch.from_numpy(observed_np)
    for update in range(updates):
        rows = rng.choice(len(feature), size=min(batch_size, len(feature)), replace=False)
        x = feature_tensor[rows].to(device, non_blocking=True)
        y = residual_tensor[rows].to(device, non_blocking=True)
        observed = observed_tensor[rows].to(device, non_blocking=True)
        distance_batch = torch.abs(y - center[None, :])
        neutral = observed & distance_batch.le(tau[None, :])
        active = observed & distance_batch.ge(2.0 * tau[None, :])
        grey = observed & ~(neutral | active)
        optimizer.zero_grad(set_to_none=True)
        if variant == "E":
            value, logits = model(x, basis, response_basis, site_mean)
            regression = ((value - y) ** 2 * active.float()).sum() / active.sum().clamp_min(1)
            class_mask = active | neutral
            class_target = active.float()
            classification = torch.nn.functional.binary_cross_entropy_with_logits(
                logits[class_mask], class_target[class_mask]
            )
            loss = regression + 0.30 * classification
        else:
            prediction = model(x, basis, site_mean)
            weight = active.float() + 0.20 * neutral.float()
            if variant == "D":
                active_count = active.sum().clamp_min(1)
                neutral_count = neutral.sum().clamp_min(1)
                keep_probability = torch.clamp(active_count.float() / neutral_count.float(), max=1.0)
                neutral_keep = torch.rand_like(prediction).lt(keep_probability) & neutral
                weight = active.float() + 0.20 * neutral_keep.float()
            loss = (((prediction - y) ** 2) * weight).sum() / weight.sum().clamp_min(1.0)
            if variant == "F":
                def weighted_corr(dim: int) -> torch.Tensor:
                    total_weight = weight.sum(dim=dim).clamp_min(1.0)
                    y_mean = (weight * y).sum(dim=dim) / total_weight
                    p_mean = (weight * prediction).sum(dim=dim) / total_weight
                    if dim == 0:
                        y_delta = y - y_mean[None, :]
                        p_delta = prediction - p_mean[None, :]
                    else:
                        y_delta = y - y_mean[:, None]
                        p_delta = prediction - p_mean[:, None]
                    covariance = (weight * y_delta * p_delta).sum(dim=dim)
                    y_variance = (weight * y_delta.square()).sum(dim=dim)
                    p_variance = (weight * p_delta.square()).sum(dim=dim)
                    valid = (weight.sum(dim=dim) >= 5) & (y_variance > 1e-8) & (p_variance > 1e-8)
                    denominator = torch.sqrt((y_variance * p_variance).clamp_min(1e-8))
                    return (covariance[valid] / denominator[valid]).mean() if valid.any() else prediction.new_tensor(0.0)

                site_corr = weighted_corr(0)
                patient_corr = weighted_corr(1)
                loss = loss + site_corr_weight * (1.0 - site_corr) + patient_corr_weight * (1.0 - patient_corr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if (update + 1) % 64 == 0:
            print(f"update {update + 1}/{updates} loss={float(loss):.6f}", flush=True)
    return model, center_np, tau_np


@torch.no_grad()
def predict_model(
    model: torch.nn.Module,
    feature: np.ndarray,
    basis_np: np.ndarray,
    site_mean_np: np.ndarray,
    device: torch.device,
    batch_size: int,
    cutoff: float | None,
) -> np.ndarray:
    basis = torch.from_numpy(basis_np).to(device)
    response_basis = torch.from_numpy(np.abs(basis_np)).to(device)
    site_mean = torch.from_numpy(site_mean_np).to(device)
    output = np.empty((len(feature), basis_np.shape[1]), dtype=np.float32)
    for start in range(0, len(feature), batch_size):
        stop = min(start + batch_size, len(feature))
        x = torch.from_numpy(feature[start:stop]).to(device)
        if isinstance(model, HurdleModel):
            value, logits = model(x, basis, response_basis, site_mean)
            probability = torch.sigmoid(logits)
            prediction = torch.where(probability >= float(cutoff), value, torch.zeros_like(value))
        else:
            prediction = model(x, basis, site_mean)
        output[start:stop] = prediction.cpu().numpy()
    return output


def center_pair(truth: np.ndarray, prediction: np.ndarray, studies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = truth.copy()
    p = prediction.copy()
    for study in np.unique(studies):
        rows = studies == study
        observed = np.isfinite(y[rows]) & np.isfinite(p[rows])
        count = observed.sum(0)
        ym = np.divide(np.where(observed, y[rows], 0).sum(0), count, out=np.zeros(y.shape[1]), where=count > 0)
        pm = np.divide(np.where(observed, p[rows], 0).sum(0), count, out=np.zeros(p.shape[1]), where=count > 0)
        y[rows] -= ym[None, :]
        p[rows] -= pm[None, :]
    return y, p


def axis_pearson(y: np.ndarray, p: np.ndarray, axis: int) -> np.ndarray:
    observed = np.isfinite(y) & np.isfinite(p)
    count = observed.sum(axis)
    y0 = np.where(observed, y, 0.0)
    p0 = np.where(observed, p, 0.0)
    ym = np.divide(y0.sum(axis), count, out=np.zeros_like(count, dtype=float), where=count > 0)
    pm = np.divide(p0.sum(axis), count, out=np.zeros_like(count, dtype=float), where=count > 0)
    if axis == 0:
        dy = np.where(observed, y - ym[None, :], 0.0)
        dp = np.where(observed, p - pm[None, :], 0.0)
    else:
        dy = np.where(observed, y - ym[:, None], 0.0)
        dp = np.where(observed, p - pm[:, None], 0.0)
    denominator = np.sqrt((dy * dy).sum(axis) * (dp * dp).sum(axis))
    return np.divide((dy * dp).sum(axis), denominator, out=np.full(count.shape, np.nan), where=(count >= 10) & (denominator > 0))


def axis_spearman(y: np.ndarray, p: np.ndarray, axis: int) -> np.ndarray:
    size = y.shape[1] if axis == 0 else y.shape[0]
    out = np.full(size, np.nan)
    for index in range(size):
        yy = y[:, index] if axis == 0 else y[index]
        pp = p[:, index] if axis == 0 else p[index]
        observed = np.isfinite(yy) & np.isfinite(pp)
        if observed.sum() >= 10:
            out[index] = np.corrcoef(rankdata(yy[observed]), rankdata(pp[observed]))[0, 1]
    return out


def metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    fixed: np.ndarray,
    studies: np.ndarray,
    center: np.ndarray,
    tau: np.ndarray,
) -> dict[str, float]:
    y, p = center_pair(truth, prediction, studies)
    patient_pearson = axis_pearson(y, p, 1)
    site_pearson = axis_pearson(y, p, 0)
    patient_spearman = axis_spearman(y, p, 1)
    site_spearman = axis_spearman(y, p, 0)
    residual_truth = truth - fixed
    residual_prediction = prediction - fixed
    distance = np.abs(residual_truth - center[None, :])
    active = np.isfinite(residual_truth) & np.isfinite(residual_prediction) & (distance >= 2.0 * tau[None, :])
    if active.sum() > 1:
        active_pearson = float(np.corrcoef(residual_truth[active], residual_prediction[active])[0, 1])
        direction = float(np.mean(np.sign(residual_truth[active]) == np.sign(residual_prediction[active])))
    else:
        active_pearson = np.nan
        direction = np.nan
    observed = np.isfinite(y) & np.isfinite(p)
    truth_sd = np.sqrt(np.divide(np.where(observed, y * y, 0).sum(0), observed.sum(0), out=np.full(y.shape[1], np.nan), where=observed.sum(0) > 1))
    pred_sd = np.sqrt(np.divide(np.where(observed, p * p, 0).sum(0), observed.sum(0), out=np.full(p.shape[1], np.nan), where=observed.sum(0) > 1))
    sd_ratio = np.divide(pred_sd, truth_sd, out=np.full_like(pred_sd, np.nan), where=truth_sd > 1e-8)
    return {
        "patient_pearson": float(np.nanmedian(patient_pearson)),
        "patient_spearman": float(np.nanmedian(patient_spearman)),
        "site_pearson": float(np.nanmedian(site_pearson)),
        "site_spearman": float(np.nanmedian(site_spearman)),
        "active_only_pearson": active_pearson,
        "active_direction_accuracy": direction,
        "median_site_sd_ratio": float(np.nanmedian(sd_ratio)),
    }


def main() -> int:
    args = arguments()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions").mkdir(exist_ok=True)
    input_dir = args.project_root / "01_data/bulk/intermediate/random70_reference_rna_protein_logscale_phosphosite_inputs_20260815"
    latent_dir = args.project_root / "01_data/multi_omics/intermediate/scp682_v2_m2_residual_latent_20260823/models"
    manifest_path = args.project_root / "01_data/multi_omics/processed/pancancer_multi_task_locked_v2/residual_target_manifest.tsv"
    rna_frame = pd.read_parquet(input_dir / "rna_reference_quantile.parquet")
    protein_frame = pd.read_parquet(input_dir / "protein_prediction.parquet").reindex(rna_frame.index)
    truth_frame = pd.read_parquet(input_dir / "phosphosite_logscale_aligned.parquet").reindex(rna_frame.index)
    split = pd.read_csv(input_dir / "split_manifest.tsv", sep="\t").set_index("sample_id").reindex(rna_frame.index)
    metadata = pd.read_csv(input_dir / "sample_metadata.tsv", sep="\t").set_index("sample_id").reindex(rna_frame.index)
    train_rows = np.flatnonzero(split.role.eq("selection_train"))
    dev_rows = np.flatnonzero(split.role.eq("selection_validation"))
    studies = metadata.study.astype(str).to_numpy()
    truth = truth_frame.to_numpy(np.float32)
    protein = protein_frame.to_numpy(np.float32)
    manifest = pd.read_csv(manifest_path, sep="\t")
    protein_lookup = {gene.upper(): index for index, gene in enumerate(protein_frame.columns.astype(str))}
    parent_index = np.asarray([protein_lookup[str(gene).upper()] for gene in manifest.total_protein_gene], dtype=np.int64)
    fixed = fit_fixed_offset(truth, protein, studies, train_rows, parent_index)
    residual = truth - fixed
    package = np.load(latent_dir / "input_projection_and_ridge.npz")
    basis = np.load(latent_dir / "site_basis_rank128.npy").astype(np.float32)
    site_mean = np.load(latent_dir / "site_mean.npy").astype(np.float32)
    feature = transform_inputs(rna_frame.to_numpy(np.float32), protein, package)
    device = torch.device(args.device)
    oof = np.empty((len(train_rows), truth.shape[1]), dtype=np.float32)
    oof_probability_cutoffs: dict[float, np.ndarray] = {}
    if args.variant == "E":
        for cutoff in (0.4, 0.5, 0.6):
            oof_probability_cutoffs[cutoff] = np.empty_like(oof)
    folds = KFold(n_splits=5, shuffle=True, random_state=20260823)
    for fold, (fit_local, held_local) in enumerate(folds.split(train_rows)):
        fit = train_rows[fit_local]
        held = train_rows[held_local]
        model, _, _ = train_model(
            args.variant, feature[fit], residual[fit], package, basis, site_mean,
            args.updates, args.batch_size, args.learning_rate,
            args.site_corr_weight, args.patient_corr_weight, device, SEED + fold,
        )
        if args.variant == "E":
            for cutoff in oof_probability_cutoffs:
                oof_probability_cutoffs[cutoff][held_local] = predict_model(
                    model, feature[held], basis, site_mean, device, args.batch_size, cutoff
                )
        else:
            oof[held_local] = predict_model(model, feature[held], basis, site_mean, device, args.batch_size, None)
        print(f"completed fold {fold + 1}/5", flush=True)
    full_center, full_tau = empirical_tau(residual[train_rows])
    if args.variant == "E":
        cutoff_rows = []
        for cutoff, residual_prediction in oof_probability_cutoffs.items():
            total = fixed[train_rows] + residual_prediction
            row = {"cutoff": cutoff, **metrics(truth[train_rows], total, fixed[train_rows], studies[train_rows], full_center, full_tau)}
            cutoff_rows.append(row)
        cutoff_table = pd.DataFrame(cutoff_rows).sort_values(["site_pearson", "patient_pearson"], ascending=False)
        selected_cutoff = float(cutoff_table.iloc[0].cutoff)
        oof = oof_probability_cutoffs[selected_cutoff]
        cutoff_table.to_csv(args.output_dir / "cutoff_grid.tsv", sep="\t", index=False)
    else:
        selected_cutoff = None
    full_model, _, _ = train_model(
        args.variant, feature[train_rows], residual[train_rows], package, basis, site_mean,
        args.updates, args.batch_size, args.learning_rate,
        args.site_corr_weight, args.patient_corr_weight, device, SEED + 100,
    )
    dev_residual_prediction = predict_model(
        full_model, feature[dev_rows], basis, site_mean, device, args.batch_size, selected_cutoff
    )
    train_total = fixed[train_rows] + oof
    dev_total = fixed[dev_rows] + dev_residual_prediction
    rows = [
        {"scope": "train_1796_OOF", "variant": args.variant, "cutoff": selected_cutoff, **metrics(
            truth[train_rows], train_total, fixed[train_rows], studies[train_rows], full_center, full_tau
        )},
        {"scope": "development_770", "variant": args.variant, "cutoff": selected_cutoff, **metrics(
            truth[dev_rows], dev_total, fixed[dev_rows], studies[dev_rows], full_center, full_tau
        )},
    ]
    pd.DataFrame(rows).to_csv(args.output_dir / "metrics.tsv", sep="\t", index=False)
    np.savez_compressed(
        args.output_dir / "predictions/predictions.npz",
        train_ids=truth_frame.index[train_rows].astype(str).to_numpy(),
        dev_ids=truth_frame.index[dev_rows].astype(str).to_numpy(),
        site_ids=truth_frame.columns.astype(str).to_numpy(),
        train_oof=train_total,
        dev=dev_total,
    )
    torch.save(full_model.state_dict(), args.output_dir / "model.pt")
    report = {
        "status": "complete",
        "variant": args.variant,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "site_corr_weight": args.site_corr_weight,
        "patient_corr_weight": args.patient_corr_weight,
        "tau_median": float(np.median(full_tau)),
        "selected_cutoff": selected_cutoff,
        "metrics": rows,
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "SUCCESS").touch()
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
