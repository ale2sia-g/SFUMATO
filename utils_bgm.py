"""Helper functions for GPU BGM clustering and visualization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import dendrogram, fcluster
from scipy.spatial.distance import pdist
from sklearn.preprocessing import normalize as sk_normalize

from utils_config import default_results_dir, resolve_path
from utils_preprocess import (
    make_bgm_stem,
    make_preprocess_stem,
    make_shared_bgm_stem,
    p2r_cache_keys,
)


@dataclass(frozen=True)
class BGMConfig:
    run_name: str
    k_list: list[int]
    bgm_oversample: float
    bin_width: int
    factor: int
    seed: int
    comp: int
    save_p2r: bool
    cache_dir: Path
    outroot: Path
    save_transcript_proba: bool = False
    merge_method: str = "complete"
    merge_metric: str = "cosine"
    bgm_weight_prior: float = 1.0
    color_colormap: str = "gist_ncar"
    color_colormap_start: float = 0.08
    color_colormap_end: float = 0.92
    dendrogram_branch_linewidth: float = 3.0

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "BGMConfig":
        preprocess = config.get("preprocess", {})
        bgm = config.get("bgm", {})
        color = config.get("color", {})

        cache_dir = resolve_path(config, preprocess["cache_dir"])
        outroot_value = bgm.get("outroot")
        outroot = (
            resolve_path(config, outroot_value)
            if outroot_value
            else default_results_dir(cache_dir)
        )

        return cls(
            run_name=config["run_name"],
            k_list=[int(k) for k in preprocess["k_list"]],
            bgm_oversample=float(preprocess.get("bgm_oversample", 1.5)),
            bin_width=int(preprocess["bin_width"]),
            factor=int(preprocess["factor"]),
            seed=int(preprocess.get("seed", 8)),
            comp=int(preprocess["comp"]),
            save_p2r=bool(preprocess.get("save_p2r", True)),
            cache_dir=cache_dir,
            outroot=outroot,
            save_transcript_proba=bool(bgm.get("save_transcript_proba", False)),
            merge_method=str(bgm.get("merge_method", "complete")),
            merge_metric=str(bgm.get("merge_metric", "cosine")),
            bgm_weight_prior=float(bgm.get("bgm_weight_prior", 1.0)),
            color_colormap=str(color.get("colormap", "gist_ncar")),
            color_colormap_start=float(color.get("colormap_start", 0.08)),
            color_colormap_end=float(color.get("colormap_end", 0.92)),
            dendrogram_branch_linewidth=float(
                color.get("dendrogram_branch_linewidth", 3.0)
            ),
        )


def result_dir_for(cfg: BGMConfig, k: int) -> Path:
    return cfg.outroot / f"K{k}" / f"SVD{cfg.comp}"


def image_dir_for(cfg: BGMConfig, k: int) -> Path:
    return result_dir_for(cfg, k) / "images"


def shared_result_dir_for(cfg: BGMConfig) -> Path:
    return cfg.outroot / "shared_BGM" / f"SVD{cfg.comp}"


def shared_image_dir_for(cfg: BGMConfig) -> Path:
    return shared_result_dir_for(cfg) / "images"


def cache_file_for(cfg: BGMConfig) -> Path:
    stem = make_preprocess_stem(
        cfg.run_name,
        cfg.bin_width,
        cfg.factor,
        cfg.comp,
    )
    return cfg.cache_dir / f"{stem}.npz"


def bgm_stem_for(cfg: BGMConfig, k: int, k_bgm: int) -> str:
    return make_bgm_stem(
        cfg.run_name,
        k_bgm,
        k,
        cfg.bin_width,
        cfg.factor,
        cfg.comp,
        cfg.bgm_oversample,
    )


def shared_bgm_stem_for(cfg: BGMConfig, k_bgm: int, k_max: int) -> str:
    return make_shared_bgm_stem(
        cfg.run_name,
        k_bgm,
        k_max,
        cfg.bin_width,
        cfg.factor,
        cfg.comp,
    )


def rgb01_to_hex(rgb01: np.ndarray) -> str:
    rgb = np.clip(np.asarray(rgb01), 0.0, 1.0)
    rgb255 = (rgb * 255).astype(int)
    return "#{:02x}{:02x}{:02x}".format(rgb255[0], rgb255[1], rgb255[2])


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
    X: np.ndarray,
    proba: np.ndarray,
    eps: float = 1e-12,
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
    proba: np.ndarray,
    merge_labels: np.ndarray,
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
            adata_o.obs[col] = (
                adata_o.obs[col].replace("NaN", "#000000").fillna("#000000")
            )

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


# ==========================
# HIERARCHICAL COLORS
# ==========================

def get_continuous_colormap_func(
    cmap_name: str = "gist_ncar",
    start: float = 0.08,
    end: float = 0.92,
) -> Callable[[float], np.ndarray]:
    import matplotlib as mpl

    cmap = mpl.colormaps[cmap_name]
    start = float(start)
    end = float(end)

    if not (0.0 <= start <= 1.0 and 0.0 <= end <= 1.0):
        raise ValueError("Colormap start/end must be within [0, 1].")
    if start >= end:
        raise ValueError("Colormap start must be smaller than colormap end.")

    def color_func(t: float) -> np.ndarray:
        t = float(np.clip(t, 0.0, 1.0))
        return np.asarray(cmap(start + t * (end - start))[:3], dtype=float)

    return color_func


def leaf_rank_from_linkage(Z: np.ndarray) -> dict[int, int]:
    d = dendrogram(Z, no_plot=True)
    leaf_order = d["leaves"]
    return {int(leaf): int(rank) for rank, leaf in enumerate(leaf_order)}


def node_members_from_linkage(Z: np.ndarray) -> dict[int, set[int]]:
    n_leaves = Z.shape[0] + 1
    node_members = {i: {i} for i in range(n_leaves)}

    for row_idx, row in enumerate(Z):
        left, right = int(row[0]), int(row[1])
        node_id = n_leaves + row_idx
        node_members[node_id] = node_members[left] | node_members[right]

    return node_members


def remap_labels_zero_based(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    mapping = {old: new for new, old in enumerate(unique_labels)}
    return np.array([mapping[x] for x in labels], dtype=np.int32)


def cluster_spans_for_cut(
    Z: np.ndarray,
    k: int,
    leaf_rank: dict[int, int],
) -> tuple[np.ndarray, dict[int, dict[str, Any]]]:
    labels_raw = fcluster(Z, t=int(k), criterion="maxclust")
    labels = remap_labels_zero_based(labels_raw)
    spans: dict[int, dict[str, Any]] = {}

    for cluster_id in np.unique(labels):
        members = np.where(labels == cluster_id)[0]
        ranks = np.array([leaf_rank[int(m)] for m in members])

        spans[int(cluster_id)] = {
            "members": members,
            "rank_min": int(ranks.min()),
            "rank_max": int(ranks.max()),
            "rank_center": float((ranks.min() + ranks.max()) / 2),
        }

    return labels, spans


def colors_for_cut(
    Z: np.ndarray,
    k: int,
    leaf_rank: dict[int, int],
    color_func: Callable[[float], np.ndarray],
) -> tuple[np.ndarray, dict[int, dict[str, Any]], dict[int, str], dict[int, np.ndarray]]:
    labels, spans = cluster_spans_for_cut(Z, k, leaf_rank)
    n_leaves = Z.shape[0] + 1

    colors_hex: dict[int, str] = {}
    colors_rgb: dict[int, np.ndarray] = {}

    for cluster_id, info in spans.items():
        t = info["rank_center"] / max(n_leaves - 1, 1)
        rgb = color_func(t)
        colors_rgb[int(cluster_id)] = rgb
        colors_hex[int(cluster_id)] = rgb01_to_hex(rgb)

    return labels, spans, colors_hex, colors_rgb


def cut_height_for_k(Z: np.ndarray, k: int) -> float | None:
    n_leaves = Z.shape[0] + 1
    k = int(k)

    if k >= n_leaves or k <= 1:
        return None

    lower_idx = n_leaves - k - 1
    upper_idx = n_leaves - k

    lower = Z[lower_idx, 2] if lower_idx >= 0 else 0.0
    upper = Z[upper_idx, 2] if upper_idx < Z.shape[0] else Z[-1, 2]

    return float((lower + upper) / 2.0)


def link_color_func_for_cut(
    labels: np.ndarray,
    cluster_colors: dict[int, str],
    node_members: dict[int, set[int]],
) -> Callable[[int], str]:
    def func(node_id: int) -> str:
        members = node_members[int(node_id)]
        cls = {int(labels[m]) for m in members}

        if len(cls) == 1:
            cluster_id = next(iter(cls))
            return cluster_colors[cluster_id]

        return "#bdbdbd"

    return func


def link_color_func_multiresolution(
    Z: np.ndarray,
    k_list: list[int],
    leaf_rank: dict[int, int],
    node_members: dict[int, set[int]],
    color_func: Callable[[float], np.ndarray],
) -> Callable[[int], str]:
    k_sorted = sorted([int(k) for k in k_list], reverse=True)

    labels_by_k = {}
    colors_by_k = {}

    for k in k_sorted:
        labels, _, colors_hex, _ = colors_for_cut(Z, k, leaf_rank, color_func)
        labels_by_k[k] = labels
        colors_by_k[k] = colors_hex

    def func(node_id: int) -> str:
        members = node_members[int(node_id)]

        for k in k_sorted:
            labels = labels_by_k[k]
            cls = {int(labels[m]) for m in members}

            if len(cls) == 1:
                cluster_id = next(iter(cls))
                return colors_by_k[k][cluster_id]

        return "#bdbdbd"

    return func


def thicken_dendrogram_lines(ax, linewidth: float = 3.0) -> None:
    for collection in ax.collections:
        try:
            collection.set_linewidth(linewidth)
        except Exception:
            pass


def plot_multiresolution_dendrogram(
    Z: np.ndarray,
    k_list: list[int],
    color_func: Callable[[float], np.ndarray],
    path: Path,
    title: str,
    branch_linewidth: float = 3.0,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)

    k_list = [int(k) for k in k_list]
    n_leaves = Z.shape[0] + 1
    leaf_rank = leaf_rank_from_linkage(Z)
    node_members = node_members_from_linkage(Z)

    fig_height = 7.5 + 0.38 * len(k_list)
    fig, ax = plt.subplots(figsize=(15, fig_height), dpi=220)

    dendro = dendrogram(
        Z,
        ax=ax,
        labels=[str(i) for i in range(n_leaves)],
        leaf_rotation=90,
        leaf_font_size=6,
        link_color_func=link_color_func_multiresolution(
            Z,
            k_list,
            leaf_rank,
            node_members,
            color_func,
        ),
        above_threshold_color="#9e9e9e",
    )

    thicken_dendrogram_lines(ax, linewidth=branch_linewidth)

    displayed_leaves = dendro["leaves"]
    x_by_leaf = {int(leaf): 5 + 10 * i for i, leaf in enumerate(displayed_leaves)}

    _, ymax = ax.get_ylim()
    strip_h = 0.045 * ymax
    gap = 0.016 * ymax
    base_y = -strip_h - gap

    for k in k_list:
        h = cut_height_for_k(Z, k)
        if h is None:
            continue

        ax.axhline(
            h,
            color="#333333",
            linewidth=1.2,
            linestyle="--",
            alpha=0.75,
        )
        ax.text(
            ax.get_xlim()[1] + 5,
            h,
            f"K={k}",
            va="center",
            ha="left",
            fontsize=8,
            color="#333333",
        )

    for row, k in enumerate(k_list):
        _, spans, colors_hex, _ = colors_for_cut(Z, k, leaf_rank, color_func)
        y = base_y - row * (strip_h + gap)

        ax.text(
            -12,
            y + strip_h / 2,
            f"K={k}",
            va="center",
            ha="right",
            fontsize=9,
        )

        for cluster_id, info in spans.items():
            members = info["members"]
            xs = np.array([x_by_leaf[int(m)] for m in members])
            x0 = xs.min() - 5
            width = xs.max() - xs.min() + 10

            ax.add_patch(
                plt.Rectangle(
                    (x0, y),
                    width,
                    strip_h,
                    facecolor=colors_hex[int(cluster_id)],
                    edgecolor="white",
                    linewidth=0.5,
                    clip_on=False,
                )
            )

    ax.set_ylim(base_y - len(k_list) * (strip_h + gap) - gap, ymax)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel("linkage distance")
    ax.set_xlabel("oversampled BGM components")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_cut_dendrogram(
    Z: np.ndarray,
    k: int,
    color_func: Callable[[float], np.ndarray],
    path: Path,
    title: str,
    branch_linewidth: float = 3.0,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)

    k = int(k)
    n_leaves = Z.shape[0] + 1
    leaf_rank = leaf_rank_from_linkage(Z)
    node_members = node_members_from_linkage(Z)

    labels, spans, colors_hex, _ = colors_for_cut(Z, k, leaf_rank, color_func)

    fig, ax = plt.subplots(figsize=(15, 6.8), dpi=220)

    dendro = dendrogram(
        Z,
        ax=ax,
        labels=[str(i) for i in range(n_leaves)],
        leaf_rotation=90,
        leaf_font_size=6,
        link_color_func=link_color_func_for_cut(labels, colors_hex, node_members),
        above_threshold_color="#9e9e9e",
    )

    thicken_dendrogram_lines(ax, linewidth=branch_linewidth)

    displayed_leaves = dendro["leaves"]
    x_by_leaf = {int(leaf): 5 + 10 * i for i, leaf in enumerate(displayed_leaves)}

    _, ymax = ax.get_ylim()
    strip_h = 0.05 * ymax
    gap = 0.025 * ymax
    y = -strip_h - gap

    h = cut_height_for_k(Z, k)
    if h is not None:
        ax.axhline(
            h,
            color="#333333",
            linewidth=1.2,
            linestyle="--",
            alpha=0.75,
        )
        ax.text(
            ax.get_xlim()[1] + 5,
            h,
            f"K={k}",
            va="center",
            ha="left",
            fontsize=8,
            color="#333333",
        )

    ax.text(
        -12,
        y + strip_h / 2,
        f"K={k}",
        va="center",
        ha="right",
        fontsize=9,
    )

    label_y = y - 0.02 * ymax

    for cluster_id, info in spans.items():
        members = info["members"]
        xs = np.array([x_by_leaf[int(m)] for m in members])
        x0 = xs.min() - 5
        width = xs.max() - xs.min() + 10

        ax.add_patch(
            plt.Rectangle(
                (x0, y),
                width,
                strip_h,
                facecolor=colors_hex[int(cluster_id)],
                edgecolor="white",
                linewidth=0.5,
                clip_on=False,
            )
        )

        ax.text(
            x0 + width / 2,
            label_y,
            f"C{int(cluster_id)}",
            ha="center",
            va="top",
            fontsize=7,
            clip_on=False,
        )

    ax.set_ylim(y - 0.13 * ymax, ymax)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel("linkage distance")
    ax.set_xlabel("oversampled BGM components")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def cluster_color_table(
    Z: np.ndarray,
    k_list: list[int],
    color_func: Callable[[float], np.ndarray],
) -> pd.DataFrame:
    leaf_rank = leaf_rank_from_linkage(Z)
    rows = []

    for k in k_list:
        _, spans, colors_hex, _ = colors_for_cut(Z, int(k), leaf_rank, color_func)

        for cluster_id, info in spans.items():
            rows.append(
                {
                    "K": int(k),
                    "cluster": int(cluster_id),
                    "n_components": int(len(info["members"])),
                    "rank_min": int(info["rank_min"]),
                    "rank_max": int(info["rank_max"]),
                    "rank_center": float(info["rank_center"]),
                    "color_hex": colors_hex[int(cluster_id)],
                    "components": ",".join(str(int(x)) for x in info["members"]),
                }
            )

    return pd.DataFrame(rows)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)