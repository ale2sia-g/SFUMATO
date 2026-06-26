#!/usr/bin/env python3
"""CPU preprocessing pipeline

This script is dataset-agnostic. All run-specific values are read from a
shared config file so the same config can be used by the CPU preprocessing
job, the GPU BGM job, and local single-process runs.
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

from utils_config import load_config
from utils_preprocess import (
    PreprocessConfig,
    make_preprocess_stem,
    p2r_cache_keys,
    resolve_rare_genes,
    save_json,
    to_dense_float32,
    tsne_colors_p2r,
    validate_and_standardize_dataframe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to JSON/YAML config.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing preprocessing cache.",
    )
    return parser.parse_args()


def run_preprocess(config: dict, force: bool = False) -> None:
    from soa_points2regions_withsplit import points2regions_withsplit

    cfg = PreprocessConfig.from_dict(config)
    if not cfg.use_svd:
        raise ValueError("Only sparse TruncatedSVD mode is currently supported.")

    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    stem = make_preprocess_stem(
        cfg.run_name, cfg.bin_width, cfg.factor, cfg.comp, cfg.use_svd
    )
    cache_file = cfg.cache_dir / f"{stem}.npz"

    if cache_file.exists() and not force:
        print(f"{cache_file} exists, skipping. Use --force to overwrite.", flush=True)
        return

    print(f"Reading input CSV: {cfg.input_csv}", flush=True)
    df_raw = pd.read_csv(cfg.input_csv)
    df = validate_and_standardize_dataframe(df_raw, cfg)
    gene_positions = df[["x", "y"]].to_numpy()
    rare_gene_list = resolve_rare_genes(df, cfg.rare_genes)

    print(f"There are {df.target_name.nunique()} gene types", flush=True)
    print(f"Rare genes used by points2regions: {rare_gene_list}", flush=True)
    print("Using sparse TruncatedSVD instead of dense PCA.", flush=True)
    print("Running soa_points2regions_withsplit once for this run...", flush=True)

    res = points2regions_withsplit(
        xy=gene_positions,
        labels=df["target_name"].to_numpy(),
        bin_width=cfg.bin_width,
        smooth=cfg.factor,
        rare_genes=rare_gene_list,
        min_genes_per_bin=cfg.min_genes_per_bin,
        alpha=cfg.alpha,
    )

    B = res["bin_matrix"]
    back_map = B.T.nonzero()[1]
    xy_bin = res["xy_bin"]
    good_bins = res["good_bins"]
    good_bin_ids = np.where(good_bins)[0]
    pos = xy_bin[good_bins]

    X_all_sparse = res["X_all"][good_bins].astype(np.float32)
    X_rare = to_dense_float32(res["X_rare"][good_bins])

    print(f"  Good bins: {good_bins.sum():,} / {len(good_bins):,}", flush=True)
    print(
        f"  X_all sparse: {X_all_sparse.shape}, nnz={X_all_sparse.nnz:,}",
        flush=True,
    )
    print(f"  X_rare: {X_rare.shape}, {X_rare.nbytes / 1e6:.1f} MB", flush=True)

    df_run = df.copy()
    df_run["bin_id"] = back_map

    print(f"TruncatedSVD ({cfg.comp} components)...", flush=True)
    svd_bgm = TruncatedSVD(n_components=cfg.comp, random_state=cfg.seed)
    X_norm = svd_bgm.fit_transform(X_all_sparse).astype(np.float32)
    print(
        f"  Explained variance: {svd_bgm.explained_variance_ratio_.round(3)}",
        flush=True,
    )
    print(f"  Cumulative: {svd_bgm.explained_variance_ratio_.sum():.3f}", flush=True)

    del X_all_sparse, svd_bgm
    gc.collect()

    save_dict = dict(
        X_norm=X_norm,
        X_rare=X_rare,
        good_bin_ids=good_bin_ids.astype(np.int32),
        pos=pos.astype(np.float32),
        back_map=back_map.astype(np.int32),
        gene_x=df_run["x"].to_numpy().astype(np.int32),
        gene_y=df_run["y"].to_numpy().astype(np.int32),
        gene_name=df_run["target_name"].to_numpy(),
        gene_bin_id=df_run["bin_id"].to_numpy().astype(np.int32),
        n_bins_total=np.array([len(good_bins)], dtype=np.int32),
        k_list=np.asarray(cfg.k_list, dtype=np.int32),
        bgm_oversample=np.array([cfg.bgm_oversample], dtype=np.float32),
    )

    if cfg.save_p2r:
        X_p2r = res["X_all"]
        for k in cfg.k_list:
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

            cluster_key, color_key = p2r_cache_keys(k_p2r)
            save_dict[cluster_key] = cluster_p2r_bins
            save_dict[color_key] = p2r_color_bins
            print(f"  P2R clusters for K={k_p2r}: {n_p2r} unique", flush=True)

            del km, Z_p2r, merged_p2r, cluster_p2r_bins, p2r_color_bins
            del p2r_centers, p2r_color_list, km_labels
            gc.collect()

        del X_p2r
        gc.collect()

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
            "cache_file": str(cache_file),
            "rare_gene_list_used": rare_gene_list,
            "n_transcripts": int(len(df_run)),
            "n_genes": int(df_run["target_name"].nunique()),
            "n_good_bins": int(len(good_bin_ids)),
        },
    )

    print(f"  Saved {cache_file}", flush=True)
    print(f"  X_norm: {X_norm.shape}, {X_norm.nbytes / 1e6:.1f} MB", flush=True)
    print(f"  X_rare: {X_rare.shape}, {X_rare.nbytes / 1e6:.1f} MB", flush=True)

    del res, X_norm, X_rare, save_dict, df_run, df_raw, df
    gc.collect()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_preprocess(config, force=args.force)


if __name__ == "__main__":
    main()
