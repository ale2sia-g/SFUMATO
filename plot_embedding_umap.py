#!/usr/bin/env python3
"""Plot UMAP representations from the SFUMATO SVD cache and BGM bin CSVs."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils_bgm import (
    BGMConfig,
    bgm_stem_for,
    cache_file_for,
    image_dir_for,
    result_dir_for,
)
from utils_config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to JSON/YAML config.")
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Optional single K to plot. If omitted, all K values are plotted.",
    )
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--umap-components", type=int, default=None)
    parser.add_argument("--n-neighbors", type=int, default=None)
    parser.add_argument("--min-dist", type=float, default=None)
    parser.add_argument("--point-size", type=float, default=None)
    return parser.parse_args()


def load_umap():
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "umap-learn is required for UMAP plots. Install umap-learn in the "
            "environment or skip this optional post-processing step."
        ) from exc
    return umap


def shared_k_bgm(cfg: BGMConfig) -> int:
    return int(np.ceil(cfg.bgm_oversample * max(int(k) for k in cfg.k_list)))


def read_bin_metadata(bin_csv: Path | None) -> pd.DataFrame | None:
    if bin_csv is None or not bin_csv.exists():
        print("No matching bin-level CSV found. Plotting UMAP without cluster colors.")
        return None

    print(f"Reading bin metadata: {bin_csv}", flush=True)
    cols_try = [
        "bin_id",
        "cluster",
        "color_hard_hsv",
        "color_log_hsv",
        "color_p2r",
        "p1",
        "compl_p1",
        "x",
        "y",
    ]
    header = pd.read_csv(bin_csv, nrows=0).columns.tolist()
    usecols = [c for c in cols_try if c in header]
    return pd.read_csv(bin_csv, usecols=usecols)


def choose_indices(
    n: int,
    max_points: int,
    seed: int,
    meta: pd.DataFrame | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if n <= max_points:
        return np.arange(n)

    if meta is not None and "cluster" in meta.columns:
        idx_parts = []
        cluster_values = meta["cluster"].to_numpy()
        clusters = pd.unique(meta["cluster"].dropna())
        per_cluster_min = 500
        base = max(per_cluster_min, max_points // max(len(clusters), 1))

        for cluster in clusters:
            idx_c = np.where(cluster_values == cluster)[0]
            if len(idx_c) == 0:
                continue
            keep = min(len(idx_c), base)
            idx_parts.append(rng.choice(idx_c, size=keep, replace=False))

        if idx_parts:
            idx = np.unique(np.concatenate(idx_parts))
            if len(idx) > max_points:
                idx = rng.choice(idx, size=max_points, replace=False)
            elif len(idx) < max_points:
                remaining = np.setdiff1d(np.arange(n), idx, assume_unique=False)
                add_n = min(len(remaining), max_points - len(idx))
                if add_n > 0:
                    idx = np.concatenate(
                        [idx, rng.choice(remaining, size=add_n, replace=False)]
                    )
            return np.sort(idx)

    return np.sort(rng.choice(n, size=max_points, replace=False))


def save_scatter(
    path: Path,
    emb2d: np.ndarray,
    colors: np.ndarray | None = None,
    values: np.ndarray | None = None,
    title: str = "",
    cmap: str = "viridis",
    point_size: float = 0.35,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7), dpi=220)

    if colors is not None:
        ax.scatter(
            emb2d[:, 0],
            emb2d[:, 1],
            c=colors,
            s=point_size,
            linewidths=0,
            alpha=0.85,
        )
    elif values is not None:
        sc = ax.scatter(
            emb2d[:, 0],
            emb2d[:, 1],
            c=values,
            s=point_size,
            linewidths=0,
            alpha=0.85,
            cmap=cmap,
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
    else:
        ax.scatter(emb2d[:, 0], emb2d[:, 1], s=point_size, linewidths=0, alpha=0.85)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_for_k(
    cfg: BGMConfig,
    k: int,
    k_bgm: int,
    X_norm: np.ndarray,
    good_bin_ids: np.ndarray,
    pos: np.ndarray,
    umap_settings: dict,
) -> None:
    outdir = result_dir_for(cfg, k)
    image_dir = image_dir_for(cfg, k)
    image_dir.mkdir(parents=True, exist_ok=True)

    stem = bgm_stem_for(cfg, k, k_bgm)
    bin_csv = outdir / f"{stem}_binlevel_FULL.csv"

    meta = read_bin_metadata(bin_csv)
    if meta is not None:
        meta = meta.set_index("bin_id").reindex(good_bin_ids).reset_index()

    max_points = int(umap_settings.get("max_points", 200000))
    idx = choose_indices(X_norm.shape[0], max_points, cfg.seed, meta)
    print(f"K={k}: using {len(idx):,} / {X_norm.shape[0]:,} bins for UMAP", flush=True)

    X_sub = X_norm[idx]
    pos_sub = pos[idx]
    ids_sub = good_bin_ids[idx]
    meta_sub = meta.iloc[idx].reset_index(drop=True) if meta is not None else None

    umap = load_umap()
    reducer = umap.UMAP(
        n_components=int(umap_settings.get("umap_components", 3)),
        n_neighbors=int(umap_settings.get("n_neighbors", 30)),
        min_dist=float(umap_settings.get("min_dist", 0.15)),
        metric="euclidean",
        random_state=cfg.seed,
        verbose=True,
        low_memory=True,
    )
    emb = reducer.fit_transform(X_sub)
    emb2d = emb[:, :2]

    out_dict = {
        "bin_id": ids_sub,
        "UMAP1": emb[:, 0],
        "UMAP2": emb[:, 1],
        "x": pos_sub[:, 0],
        "y": pos_sub[:, 1],
    }
    if emb.shape[1] >= 3:
        out_dict["UMAP3"] = emb[:, 2]

    out = pd.DataFrame(out_dict)
    if meta_sub is not None:
        for col in meta_sub.columns:
            if col not in out.columns:
                out[col] = meta_sub[col].to_numpy()

    csv_out = outdir / f"{stem}_umap_sampled_bins.csv"
    out.to_csv(csv_out, index=False)
    print(f"Saved: {csv_out}", flush=True)

    point_size = float(umap_settings.get("point_size", 0.35))

    save_scatter(
        image_dir / f"{stem}_umap_plain.png",
        emb2d,
        title=f"{stem}: UMAP",
        point_size=point_size,
    )

    if emb.shape[1] >= 3:
        save_scatter(
            image_dir / f"{stem}_umap_colored_by_UMAP3.png",
            emb2d,
            values=emb[:, 2],
            title=f"{stem}: UMAP colored by UMAP3",
            cmap="viridis",
            point_size=point_size,
        )

    save_scatter(
        image_dir / f"{stem}_umap_spatial_x.png",
        emb2d,
        values=pos_sub[:, 0],
        title=f"{stem}: UMAP colored by spatial x",
        cmap="viridis",
        point_size=point_size,
    )

    save_scatter(
        image_dir / f"{stem}_umap_spatial_y.png",
        emb2d,
        values=pos_sub[:, 1],
        title=f"{stem}: UMAP colored by spatial y",
        cmap="viridis",
        point_size=point_size,
    )

    if meta_sub is not None:
        for col, suffix, title in [
            ("color_hard_hsv", "umap_sfumato_hard_colors", "SFUMATO hard colors"),
            ("color_log_hsv", "umap_sfumato_mixed_colors", "SFUMATO mixed colors"),
            ("color_p2r", "umap_color_p2r", "P2R colors"),
        ]:
            if col in meta_sub.columns:
                colors = meta_sub[col].fillna("#aaaaaa").astype(str).to_numpy()
                save_scatter(
                    image_dir / f"{stem}_{suffix}.png",
                    emb2d,
                    colors=colors,
                    title=f"{stem}: UMAP colored by {title}",
                    point_size=point_size,
                )

        if "p1" in meta_sub.columns:
            save_scatter(
                image_dir / f"{stem}_umap_confidence_p1.png",
                emb2d,
                values=meta_sub["p1"].to_numpy(float),
                title=f"{stem}: UMAP colored by p1",
                cmap="magma",
                point_size=point_size,
            )
        elif "compl_p1" in meta_sub.columns:
            confidence = 1 - meta_sub["compl_p1"].to_numpy(float)
            save_scatter(
                image_dir / f"{stem}_umap_confidence_1_minus_compl_p1.png",
                emb2d,
                values=confidence,
                title=f"{stem}: UMAP colored by confidence",
                cmap="magma",
                point_size=point_size,
            )

        if "cluster" in meta_sub.columns:
            save_scatter(
                image_dir / f"{stem}_umap_cluster_id.png",
                emb2d,
                values=meta_sub["cluster"].to_numpy(float),
                title=f"{stem}: UMAP colored by cluster id",
                cmap="tab20",
                point_size=point_size,
            )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    cfg = BGMConfig.from_dict(config)

    umap_settings = dict(config.get("umap", {}))
    for key in ["max_points", "umap_components", "n_neighbors", "min_dist", "point_size"]:
        value = getattr(args, key.replace("-", "_"), None)
        if value is not None:
            umap_settings[key] = value

    cache_file = cache_file_for(cfg)
    if not cache_file.exists():
        raise FileNotFoundError(f"Cache not found: {cache_file}")

    print(f"Loading SVD cache: {cache_file}", flush=True)
    data = np.load(cache_file, allow_pickle=True)
    X_norm = data["X_norm"].astype(np.float32, copy=False)
    good_bin_ids = data["good_bin_ids"].astype(np.int64)
    pos = data["pos"].astype(np.float32, copy=False)

    k_values = [args.k] if args.k is not None else sorted([int(k) for k in cfg.k_list])
    k_bgm = shared_k_bgm(cfg)

    for k in k_values:
        plot_for_k(cfg, int(k), k_bgm, X_norm, good_bin_ids, pos, umap_settings)

    data.close()
    del data, X_norm, good_bin_ids, pos
    gc.collect()

    print("Done.", flush=True)


if __name__ == "__main__":
    main()