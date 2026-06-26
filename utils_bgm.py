"""Helper functions for GPU BGM clustering and visualization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.preprocessing import normalize as sk_normalize

from utils_config import default_results_dir, resolve_path
from utils_preprocess import make_bgm_stem, make_preprocess_stem, p2r_cache_keys


@dataclass(frozen=True)
class BGMConfig:
    run_name: str
    k_list: list[int]
    bgm_oversample: float
    bin_width: int
    factor: int
    seed: int
    use_svd: bool
    comp: int
    save_p2r: bool
    cache_dir: Path
    outroot: Path
    save_transcript_proba: bool = False
    beta_rare: float = 0.0
    merge_method: str = "complete"
    merge_metric: str = "cosine"
    bgm_weight_prior: float = 1.0
    use_hist_equalization: bool = True

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "BGMConfig":
        preprocess = config.get("preprocess", {})
        bgm = config.get("bgm", {})
        cache_dir = resolve_path(config, preprocess["cache_dir"])
        outroot_value = bgm.get("outroot")
        outroot = resolve_path(config, outroot_value) if outroot_value else default_results_dir(cache_dir)
        return cls(
            run_name=config["run_name"],
            k_list=[int(k) for k in preprocess["k_list"]],
            bgm_oversample=float(preprocess.get("bgm_oversample", 1.5)),
            bin_width=int(preprocess["bin_width"]),
            factor=int(preprocess["factor"]),
            seed=int(preprocess.get("seed", 8)),
            use_svd=bool(preprocess.get("use_svd", True)),
            comp=int(preprocess["comp"]),
            save_p2r=bool(preprocess.get("save_p2r", True)),
            cache_dir=cache_dir,
            outroot=outroot,
            save_transcript_proba=bool(bgm.get("save_transcript_proba", False)),
            beta_rare=float(bgm.get("beta_rare", 0.0)),
            merge_method=str(bgm.get("merge_method", "complete")),
            merge_metric=str(bgm.get("merge_metric", "cosine")),
            bgm_weight_prior=float(bgm.get("bgm_weight_prior", 1.0)),
            use_hist_equalization=bool(bgm.get("use_hist_equalization", True)),
        )


def result_dir_for(cfg: BGMConfig, k: int) -> Path:
    return cfg.outroot / f"K{k}" / f"SVD{cfg.comp}"


def image_dir_for(cfg: BGMConfig, k: int) -> Path:
    return result_dir_for(cfg, k) / "images"


def cache_file_for(cfg: BGMConfig) -> Path:
    stem = make_preprocess_stem(
        cfg.run_name, cfg.bin_width, cfg.factor, cfg.comp, cfg.use_svd
    )
    return cfg.cache_dir / f"{stem}.npz"


def bgm_stem_for(cfg: BGMConfig, k: int) -> str:
    k_bgm = int(np.ceil(cfg.bgm_oversample * k))
    return make_bgm_stem(
        cfg.run_name,
        k_bgm,
        k,
        cfg.bin_width,
        cfg.factor,
        cfg.comp,
        cfg.use_svd,
        cfg.bgm_oversample,
    )


def rgb01_to_hex(rgb01: np.ndarray) -> str:
    rgb = np.clip(np.asarray(rgb01), 0.0, 1.0)
    rgb255 = (rgb * 255).astype(int)
    return "#{:02x}{:02x}{:02x}".format(rgb255[0], rgb255[1], rgb255[2])


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    rgb = np.zeros_like(hsv)
    for i in range(hsv.shape[0]):
        rgb[i] = mcolors.hsv_to_rgb(hsv[i])
    return rgb


def circular_hist_equalization(
    hue: np.ndarray,
    beta: float = 0.5,
    eps: float = 1e-8,
    return_steps: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    hue = np.asarray(hue, dtype=float)
    n = len(hue)
    if n <= 1:
        return (hue.copy(), hue.copy()) if return_steps else hue.copy()

    sort_idx = np.argsort(hue)
    h_sorted = hue[sort_idx]
    gaps = np.diff(np.concatenate([h_sorted, h_sorted[:1] + 1.0]))
    k_star = int(np.argmax(gaps))
    h_cut = (h_sorted[k_star] + 0.5 * gaps[k_star]) % 1.0

    h_shifted = (hue - h_cut) % 1.0
    sort_shift = np.argsort(h_shifted)
    h_shift_sorted = h_shifted[sort_shift]
    gaps_shift = np.diff(np.concatenate([h_shift_sorted, h_shift_sorted[:1] + 1.0]))
    weights = (gaps_shift + eps) ** beta

    cdf = np.zeros(n)
    for i in range(1, n):
        cdf[i] = cdf[i - 1] + weights[i - 1]
    cdf = cdf / (cdf[-1] + weights[-1] + eps)

    h_new = np.zeros_like(hue)
    h_new[sort_shift] = cdf
    return (h_new, h_shifted) if return_steps else h_new


def centroids_to_hsv_rgb(
    centroids_3d: np.ndarray,
    use_hist_equalization: bool,
    eps: float = 1e-8,
    return_hues: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y, z = centroids_3d.T
    hue_raw = np.arctan2(y, x)
    hue_raw = (hue_raw + np.pi) / (2 * np.pi)

    if use_hist_equalization:
        hue_equalized, hue_rotated = circular_hist_equalization(
            hue_raw, beta=0.5, eps=eps, return_steps=True
        )
        hue_final = hue_equalized
    else:
        hue_rotated = hue_raw.copy()
        hue_equalized = hue_raw.copy()
        hue_final = hue_raw

    sat = (z - z.min()) / (z.max() - z.min() + eps)
    sat = sat**0.5
    sat = 0.6 + 0.4 * sat
    val = 0.9 * np.ones_like(hue_raw)

    hsv = np.stack([hue_final, sat, val], axis=1)
    rgb = hsv_to_rgb(hsv)

    if return_hues:
        return rgb, hue_raw, hue_rotated, hue_equalized
    return rgb


def color_mix_top2_log(
    base1: np.ndarray,
    base2: np.ndarray,
    p1: float,
    p2: float,
    alpha: float = 50.0,
    eps: float = 1e-12,
) -> str:
    rgb1 = np.asarray(base1, dtype=float)
    rgb2 = np.asarray(base2, dtype=float)
    w1 = np.log1p(alpha * p1)
    w2 = np.log1p(alpha * p2)
    return rgb01_to_hex((rgb1 * w1 + rgb2 * w2) / (w1 + w2 + eps))


def soft_cluster_centroids(
    X: np.ndarray, proba: np.ndarray, eps: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    masses = proba.sum(axis=0) + eps
    centroids = (proba.T @ X) / masses[:, None]
    return centroids, masses


def normalize_condensed(d: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    pos = d[d > 0]
    scale = np.median(pos) if len(pos) > 0 else 1.0
    return d / max(scale, eps)


def safe_cosine_pdist(X: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    X_safe = X + eps
    X_normed = sk_normalize(X_safe, norm="l2")
    return pdist(X_normed, metric="euclidean")


def top2_from_proba(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cluster = P.argmax(axis=1)
    p1 = P.max(axis=1)
    if P.shape[1] == 1:
        return cluster, np.zeros_like(cluster), p1, np.zeros_like(p1)
    order = np.argsort(P, axis=1)
    second_cluster = order[:, -2]
    p2 = np.take_along_axis(P, second_cluster[:, None], axis=1).ravel()
    return cluster, second_cluster, p1, p2


def merge_probabilities_from_labels(
    proba: np.ndarray, merge_labels: np.ndarray
) -> tuple[np.ndarray, list[np.ndarray]]:
    merge_labels = np.asarray(merge_labels)
    uniq = np.unique(merge_labels)
    out = np.zeros((proba.shape[0], len(uniq)), dtype=float)
    groups = []
    for j, lab in enumerate(uniq):
        idx = np.where(merge_labels == lab)[0]
        out[:, j] = proba[:, idx].sum(axis=1)
        groups.append(idx)
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    out = out / row_sums
    return out, groups


def merge_centroids_by_groups(
    centroids: np.ndarray,
    masses: np.ndarray,
    groups: list[np.ndarray],
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((len(groups), centroids.shape[1]), dtype=float)
    out_masses = np.zeros(len(groups), dtype=float)
    for j, idx in enumerate(groups):
        w = masses[idx]
        out[j] = (centroids[idx] * w[:, None]).sum(axis=0) / (w.sum() + eps)
        out_masses[j] = w.sum()
    return out, out_masses


def csv_to_h5ad(df: pd.DataFrame):
    import anndata as ad

    adata_o = ad.AnnData(obs=df)
    for col in ["color_hard_hsv", "color_log_hsv", "color_p2r"]:
        if col in adata_o.obs.columns:
            adata_o.obs[col] = adata_o.obs[col].replace("NaN", "#000000").fillna("#000000")
    for col in ["x", "y"]:
        if col in adata_o.obs.columns and adata_o.obs[col].dtype == np.float64:
            adata_o.obs[col] = adata_o.obs[col].round().astype(int)
    for col in adata_o.obs.columns:
        if adata_o.obs[col].dtype == np.float64:
            adata_o.obs[col] = adata_o.obs[col].astype(np.float32)
    cols = ["PC1", "PC2", "PC3"]
    existing = [c for c in cols if c in adata_o.obs.columns]
    if existing:
        adata_o.obsm["X_pca"] = adata_o.obs[existing].to_numpy(dtype=np.float32)
        adata_o.obs.drop(columns=existing, inplace=True)
    return adata_o


def get_p2r_from_cache(data: Any, k: int) -> tuple[np.ndarray, np.ndarray]:
    cluster_key, color_key = p2r_cache_keys(k)
    if cluster_key not in data or color_key not in data:
        available = sorted(
            key.replace("cluster_p2r_bins_K", "")
            for key in data.files
            if key.startswith("cluster_p2r_bins_K")
        )
        raise KeyError(
            f"P2R output for K={k} was requested but not found in cache. "
            f"Available P2R K values: {available}"
        )
    return data[cluster_key], data[color_key]


def _cut_height_for_k(Z: np.ndarray, k: int) -> float | None:
    n_leaves = Z.shape[0] + 1
    if k >= n_leaves or k <= 1:
        return None
    lower_idx = n_leaves - k - 1
    upper_idx = n_leaves - k
    lower = Z[lower_idx, 2] if lower_idx >= 0 else 0.0
    upper = Z[upper_idx, 2] if upper_idx < Z.shape[0] else Z[-1, 2]
    return float((lower + upper) / 2.0)


def plot_oversampled_dendrogram(
    Z: np.ndarray,
    target_k: int,
    path: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5), dpi=220)
    dendrogram(Z, ax=ax, no_labels=False, color_threshold=None)
    cut_height = _cut_height_for_k(Z, target_k)
    if cut_height is not None:
        ax.axhline(cut_height, color="black", linewidth=1.2, linestyle="--")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Oversampled BGM component")
    ax.set_ylabel("Merge distance")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_final_dendrogram_with_colors(
    centroids: np.ndarray,
    colors_rgb: list[np.ndarray],
    path: Path,
    title: str,
    method: str = "complete",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    if len(centroids) <= 1:
        return

    distances = normalize_condensed(safe_cosine_pdist(centroids))
    Z = linkage(distances, method=method)

    fig, ax = plt.subplots(figsize=(max(8, len(centroids) * 0.45), 5), dpi=220)
    labels = [str(i) for i in range(len(centroids))]
    dendro = dendrogram(Z, ax=ax, labels=labels, color_threshold=None)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Final cluster")
    ax.set_ylabel("Merge distance")

    leaf_labels = dendro["ivl"]
    x_positions = ax.get_xticks()
    y_min, y_max = ax.get_ylim()
    swatch_y = y_min - 0.08 * (y_max - y_min)
    ax.set_ylim(swatch_y - 0.03 * (y_max - y_min), y_max)

    for x, lab in zip(x_positions, leaf_labels):
        cluster_id = int(lab)
        ax.scatter(
            [x],
            [swatch_y],
            s=90,
            marker="s",
            c=[colors_rgb[cluster_id]],
            edgecolors="black",
            linewidths=0.3,
            clip_on=False,
            zorder=5,
        )

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
