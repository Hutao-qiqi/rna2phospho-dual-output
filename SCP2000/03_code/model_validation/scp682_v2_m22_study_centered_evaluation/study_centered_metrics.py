#!/usr/bin/env python3
"""SCP682 跨研究评价的统一研究内中心化指标。"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def center_pair_within_study(
    truth: np.ndarray,
    prediction: np.ndarray,
    studies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """在每个研究和位点的相同成对观测患者上分别中心化真值与预测值。"""
    y = truth.astype(np.float32, copy=True)
    p = prediction.astype(np.float32, copy=True)
    study_labels = studies.astype(str)
    for study in np.unique(study_labels):
        rows = study_labels == study
        observed = np.isfinite(truth[rows]) & np.isfinite(prediction[rows])
        count = observed.sum(axis=0)
        y_center = np.divide(
            np.where(observed, truth[rows], 0.0).sum(axis=0, dtype=np.float64),
            count,
            out=np.zeros(truth.shape[1], dtype=np.float64),
            where=count > 0,
        ).astype(np.float32)
        p_center = np.divide(
            np.where(observed, prediction[rows], 0.0).sum(axis=0, dtype=np.float64),
            count,
            out=np.zeros(prediction.shape[1], dtype=np.float64),
            where=count > 0,
        ).astype(np.float32)
        y[rows] -= y_center[None, :]
        p[rows] -= p_center[None, :]
    return y, p


def observed_axis_pearson(truth: np.ndarray, prediction: np.ndarray, axis: int, minimum: int = 10) -> np.ndarray:
    observed = np.isfinite(truth) & np.isfinite(prediction)
    count = observed.sum(axis=axis)
    y = np.where(observed, truth, 0.0)
    p = np.where(observed, prediction, 0.0)
    y_mean = np.divide(y.sum(axis=axis), count, out=np.zeros_like(count, dtype=np.float64), where=count > 0)
    p_mean = np.divide(p.sum(axis=axis), count, out=np.zeros_like(count, dtype=np.float64), where=count > 0)
    if axis == 0:
        dy = np.where(observed, truth - y_mean[None, :], 0.0)
        dp = np.where(observed, prediction - p_mean[None, :], 0.0)
    else:
        dy = np.where(observed, truth - y_mean[:, None], 0.0)
        dp = np.where(observed, prediction - p_mean[:, None], 0.0)
    denominator = np.sqrt((dy * dy).sum(axis=axis) * (dp * dp).sum(axis=axis))
    return np.divide(
        (dy * dp).sum(axis=axis),
        denominator,
        out=np.full(count.shape, np.nan, dtype=np.float64),
        where=(count >= minimum) & (denominator > 0),
    )


def patient_spearman(truth: np.ndarray, prediction: np.ndarray, minimum: int = 10) -> np.ndarray:
    output = np.full(len(truth), np.nan, dtype=np.float64)
    for row in range(len(truth)):
        observed = np.isfinite(truth[row]) & np.isfinite(prediction[row])
        if observed.sum() >= minimum:
            output[row] = np.corrcoef(rankdata(truth[row, observed]), rankdata(prediction[row, observed]))[0, 1]
    return output


def site_spearman(truth: np.ndarray, prediction: np.ndarray, minimum: int = 10) -> np.ndarray:
    output = np.full(truth.shape[1], np.nan, dtype=np.float64)
    for column in range(truth.shape[1]):
        observed = np.isfinite(truth[:, column]) & np.isfinite(prediction[:, column])
        if observed.sum() >= minimum:
            output[column] = np.corrcoef(
                rankdata(truth[observed, column]), rankdata(prediction[observed, column])
            )[0, 1]
    return output


def metric_row(
    truth: np.ndarray,
    prediction: np.ndarray,
    studies: np.ndarray,
    model: str,
    scope: str,
    coordinate: str,
) -> dict[str, float | int | str]:
    if coordinate == "study_centered":
        y, p = center_pair_within_study(truth, prediction, studies)
    elif coordinate == "absolute":
        y, p = truth, prediction
    else:
        raise ValueError(coordinate)
    observed = np.isfinite(y) & np.isfinite(p)
    patient_pearson = observed_axis_pearson(y, p, axis=1)
    site_pearson = observed_axis_pearson(y, p, axis=0)
    patient_rank = patient_spearman(y, p)
    site_rank = site_spearman(y, p)
    patient_mse = np.nanmean(np.where(observed, (y - p) ** 2, np.nan), axis=1)
    return {
        "scope": scope,
        "model": model,
        "coordinate": coordinate,
        "patients": int(len(y)),
        "patient_pearson_median": float(np.nanmedian(patient_pearson)),
        "patient_pearson_mean": float(np.nanmean(patient_pearson)),
        "patient_spearman_median": float(np.nanmedian(patient_rank)),
        "patient_spearman_mean": float(np.nanmean(patient_rank)),
        "site_pearson_median": float(np.nanmedian(site_pearson)),
        "site_pearson_mean": float(np.nanmean(site_pearson)),
        "site_spearman_median": float(np.nanmedian(site_rank)),
        "site_spearman_mean": float(np.nanmean(site_rank)),
        "patient_equal_mse": float(np.nanmean(patient_mse)),
        "effective_patients": int(np.isfinite(patient_pearson).sum()),
        "effective_sites": int(np.isfinite(site_pearson).sum()),
    }


def metric_pair(truth: np.ndarray, prediction: np.ndarray, studies: np.ndarray, model: str, scope: str) -> list[dict]:
    return [
        metric_row(truth, prediction, studies, model, scope, "absolute"),
        metric_row(truth, prediction, studies, model, scope, "study_centered"),
    ]
