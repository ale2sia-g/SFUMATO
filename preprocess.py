#!/usr/bin/env python3
"""CPU preprocessing pipeline.

This script is dataset-agnostic. All run-specific values are read from a
shared config file so the same config can be used by the CPU preprocessing
job, the GPU BGM job, and local single-process runs.

The preprocessing step runs points2regions once, applies sparse TruncatedSVD,
and saves everything needed by the GPU BGM job.
"""

from __future__ import annotations

import argparse
import gc
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD

from points2regions import points2regions
from utils_config import load_config
from utils_preprocess import (
    PreprocessConfig,
    make_preprocess_stem,
    p2r_cache_keys,
    save_json,
    tsne_colors_p2r,
    validate_and_standardize_dataframe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to JSON/YAML config.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing preprocessing cache.",
    )
    return parser.parse_args()


def add_p2r_to_save_dict(
    save_dict: dict,
    p2r_results: dict[int, tuple[np.ndarray, np.ndarray]],
) -> None:
    for k, (cluster_p2r_bins, p2r_color_bins) in p2r_results.items():
        cluster_key, color_key = p2r_cache_keys(k)
        save_dict[cluster_key] = cluster_p2r_bins
        save_dict[color_key] = p2r_color_bins


def run_p2r_clustering(
    X_p2r,
    good_bins: np.ndarray,
    k_list: list[int],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    p2r_results: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for k in k_list:
        k_p2r = int(k)
        print(f"Running P2R clustering for K={k_p2r}...", flush=True)

        km = MiniBatchKMeans(
            n_clusters=int(np.ceil(1.5 * k_p2r)),
            init="k-means++",
            max_iter=100,
            batch_size=1024,
            random_state=42,
            n_init="auto",
            tol=0.0,
            max_no_improvement=10,
            reassignment_ratio=0.005,
        )
        km.fit(X_p2r[good_bins])

        Z_p2r = linkage(km.cluster_centers_, method="complete", metric="cosine")
        merged_p2r = fcluster(Z_p2r, t=k_p2r, criterion="maxclust") - 1

        cluster_p2r_bins = np.full(X_p2r.shape[0], -1, dtype=np.int32)
        km_labels = km.predict(X_p2r[good_bins])
        cluster_p2r_bins[good_bins] = merged_p2r[km_labels]

        n_p2r = len(np.unique(merged_p2r))
        p2r_centers = np.zeros((n_p2r, km.cluster_centers_.shape[1]))
        for c in range(n_p2r):
            mask_c = merged_p2r == c
            if mask_c.any():
                p2r_centers[c] = km.cluster_centers_[mask_c].mean(axis=0)

        p2r_color_list = tsne_colors_p2r(p2r_centers, seed=42)
        p2r_color_bins = np.array(["#000000"] * X_p2r.shape[0], dtype=object)
        for c, color in enumerate(p2r_color_list):
            p2r_color_bins[cluster_p2r_bins == c] = color

        p2r_results[k_p2r] = (cluster_p2r_bins, p2r_color_bins)
        print(f"  P2R clusters for K={k_p2r}: {n_p2r} unique", flush=True)

        del km, Z_p2r, merged_p2r, km_labels
        del p2r_centers, p2r_color_list
        gc.collect()

    return p2r_results


def run_preprocess(config: dict, force: bool = False) -> None:
    cfg = PreprocessConfig.from_dict(config)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    stem = make_preprocess_stem(
        cfg.run_name,
        cfg.bin_width,
        cfg.factor,
        cfg.comp,
    )
    cache_file = cfg.cache_dir / f"{stem}.npz"

    if cache_file.exists() and not force:
        print(f"{cache_file} exists, skipping. Use --force to overwrite.", flush=True)
        return

    print(f"Reading input CSV: {cfg.input_csv}", flush=True)
    df_raw = pd.read_csv(cfg.input_csv)
    df = validate_and_standardize_dataframe(df_raw, cfg)
    gene_positions = df[["x", "y"]].to_numpy()

    print(f"There are {df.target_name.nunique()} gene types", flush=True)
    print("Running points2regions once for this run...", flush=True)

    res = points2regions(
        xy=gene_positions,
        labels=df["target_name"].to_numpy(),
        bin_width=cfg.bin_width,
        smooth=cfg.factor,
        min_genes_per_bin=cfg.min_genes_per_bin,
        alpha=cfg.alpha,
    )

    B = res["bin_matrix"]
    back_map = B.T.nonzero()[1]
    xy_bin = res["xy_bin"]
    good_bins = res["good_bins"]
    good_bin_ids = np.where(good_bins)[0]
    pos = xy_bin[good_bins]
    genes_all = np.asarray(res.get("genes_all", np.array([], dtype=object)), dtype=object)

    X_all_sparse = res["X_all"][good_bins].astype(np.float32)

    print(f"  Good bins: {good_bins.sum():,} / {len(good_bins):,}", flush=True)
    print(
        f"  X_all sparse: {X_all_sparse.shape}, nnz={X_all_sparse.nnz:,}",
        flush=True,
    )

    df_run = df.copy()
    df_run["bin_id"] = back_map

    p2r_results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if cfg.save_p2r:
        p2r_results = run_p2r_clustering(
            res["X_all"],
            good_bins,
            cfg.k_list,
        )

    print(f"TruncatedSVD ({cfg.comp} components)...", flush=True)
    svd = TruncatedSVD(n_components=cfg.comp, random_state=cfg.seed)
    X_norm = svd.fit_transform(X_all_sparse).astype(np.float32)
    explained = svd.explained_variance_ratio_

    print(f"  Explained variance: {explained.round(3)}", flush=True)
    print(f"  Cumulative: {explained.sum():.3f}", flush=True)

    save_dict = dict(
        X_norm=X_norm,
        good_bin_ids=good_bin_ids.astype(np.int32),
        pos=pos.astype(np.float32),
        back_map=back_map.astype(np.int32),
        gene_x=df_run["x"].to_numpy().astype(np.int32),
        gene_y=df_run["y"].to_numpy().astype(np.int32),
        gene_name=df_run["target_name"].to_numpy(),
        gene_bin_id=df_run["bin_id"].to_numpy().astype(np.int32),
        genes_all=genes_all,
        n_bins_total=np.array([len(good_bins)], dtype=np.int32),
        k_list=np.asarray(cfg.k_list, dtype=np.int32),
        bgm_oversample=np.array([cfg.bgm_oversample], dtype=np.float32),
    )
    add_p2r_to_save_dict(save_dict, p2r_results)

    print(f"Saving cache to {cache_file}...", flush=True)
    np.savez_compressed(cache_file, **save_dict)

    save_json(
        cfg.cache_dir / f"{stem}_preprocess_config.json",
        {
            "run_name": cfg.run_name,
            "input_csv": str(cfg.input_csv),
            "columns": {
                "x": cfg.x_col,
                "y": cfg.y_col,
                "gene": cfg.gene_col,
                "num_id": cfg.num_id_col,
            },
            "preprocess": config.get("preprocess", {}),
            "reduction_method": "SVD",
            "cache_file": str(cache_file),
            "n_transcripts": int(len(df_run)),
            "n_genes": int(df_run["target_name"].nunique()),
            "n_good_bins": int(len(good_bin_ids)),
            "svd_explained_variance": explained.astype(float).tolist(),
            "svd_explained_variance_sum": float(explained.sum()),
        },
    )

    print(f"  Saved {cache_file}", flush=True)
    print(f"  X_norm: {X_norm.shape}, {X_norm.nbytes / 1e6:.1f} MB", flush=True)

    del res, X_all_sparse, X_norm, svd, save_dict, p2r_results
    del df_run, df_raw, df
    gc.collect()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_preprocess(config, force=args.force)


if __name__ == "__main__":
    main()