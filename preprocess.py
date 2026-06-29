#!/usr/bin/env python3
"""CPU preprocessing pipeline.

This script is dataset-agnostic. All run-specific values are read from a
shared config file so the same config can be used by the CPU preprocessing
job, the GPU BGM job, and local single-process runs.

The preprocessing step runs points2regions once, then writes one cache per
enabled dimensionality-reduction method: SVD, PCA, or both.
"""

from __future__ import annotations

import argparse
import gc
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA, TruncatedSVD

from utils_config import load_config
from utils_preprocess import (
    PreprocessConfig,
    make_preprocess_stem,
    method_cache_dir,
    method_label,
    method_uses_svd,
    p2r_cache_keys,
    reduction_methods_from_config,
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
        help="Overwrite existing preprocessing caches.",
    )
    return parser.parse_args()


def build_common_save_dict(
    X_rare: np.ndarray,
    rare_group_names: np.ndarray,
    rare_group_genes: dict,
    rare_group_genes_present: dict,
    genes_all: np.ndarray,
    good_bin_ids: np.ndarray,
    pos: np.ndarray,
    back_map: np.ndarray,
    df_run: pd.DataFrame,
    good_bins: np.ndarray,
    cfg: PreprocessConfig,
) -> dict:
    return dict(
        X_rare=X_rare,
        rare_group_names=np.asarray(rare_group_names, dtype=object),
        rare_group_genes_json=np.array(
            [json.dumps(rare_group_genes, sort_keys=True)],
            dtype=object,
        ),
        rare_group_genes_present_json=np.array(
            [json.dumps(rare_group_genes_present, sort_keys=True)],
            dtype=object,
        ),
        genes_all=np.asarray(genes_all, dtype=object),
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


def compute_embedding(method: str, X_all_sparse, cfg: PreprocessConfig) -> np.ndarray:
    label = method_label(method)
    print(f"{label} ({cfg.comp} components)...", flush=True)

    if method == "svd":
        reducer = TruncatedSVD(n_components=cfg.comp, random_state=cfg.seed)
        X_norm = reducer.fit_transform(X_all_sparse).astype(np.float32)
        explained = reducer.explained_variance_ratio_
    elif method == "pca":
        X_dense = to_dense_float32(X_all_sparse)
        reducer = PCA(n_components=cfg.comp, random_state=cfg.seed)
        X_norm = reducer.fit_transform(X_dense).astype(np.float32)
        explained = reducer.explained_variance_ratio_
        del X_dense
    else:
        raise ValueError(f"Unknown reduction method: {method}")

    print(f"  Explained variance: {explained.round(3)}", flush=True)
    print(f"  Cumulative: {explained.sum():.3f}", flush=True)

    del reducer
    gc.collect()
    return X_norm


def save_cache_for_method(
    method: str,
    X_norm: np.ndarray,
    common_save_dict: dict,
    p2r_results: dict[int, tuple[np.ndarray, np.ndarray]],
    cfg: PreprocessConfig,
    config: dict,
    rare_gene_groups: dict[str, list[str]],
    rare_group_names: np.ndarray,
    rare_group_genes_present: dict,
    n_transcripts: int,
    n_genes: int,
    n_good_bins: int,
    force: bool,
) -> None:
    use_svd = method_uses_svd(method)
    cache_dir = method_cache_dir(cfg.cache_dir, method)
    cache_dir.mkdir(parents=True, exist_ok=True)

    stem = make_preprocess_stem(
        cfg.run_name,
        cfg.bin_width,
        cfg.factor,
        cfg.comp,
        use_svd,
    )
    cache_file = cache_dir / f"{stem}.npz"

    if cache_file.exists() and not force:
        print(f"{cache_file} exists, skipping. Use --force to overwrite.", flush=True)
        return

    save_dict = dict(common_save_dict)
    save_dict["X_norm"] = X_norm
    add_p2r_to_save_dict(save_dict, p2r_results)

    print(f"Saving {method_label(method)} cache to {cache_file}...", flush=True)
    np.savez_compressed(cache_file, **save_dict)

    save_json(
        cache_dir / f"{stem}_preprocess_config.json",
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
            "reduction_method": method,
            "cache_file": str(cache_file),
            "rare_gene_groups_used": rare_gene_groups,
            "rare_group_names": [str(x) for x in rare_group_names],
            "rare_group_genes_present": rare_group_genes_present,
            "n_transcripts": int(n_transcripts),
            "n_genes": int(n_genes),
            "n_good_bins": int(n_good_bins),
        },
    )

    print(f"  Saved {cache_file}", flush=True)
    print(f"  X_norm: {X_norm.shape}, {X_norm.nbytes / 1e6:.1f} MB", flush=True)
    print(
        f"  X_rare: {common_save_dict['X_rare'].shape}, "
        f"{common_save_dict['X_rare'].nbytes / 1e6:.1f} MB",
        flush=True,
    )

    del save_dict
    gc.collect()


def run_preprocess(config: dict, force: bool = False) -> None:
    from soa_points2regions_withsplit import points2regions_withsplit

    cfg = PreprocessConfig.from_dict(config)
    methods = reduction_methods_from_config(cfg)

    print(f"Reading input CSV: {cfg.input_csv}", flush=True)
    df_raw = pd.read_csv(cfg.input_csv)
    df = validate_and_standardize_dataframe(df_raw, cfg)
    gene_positions = df[["x", "y"]].to_numpy()
    rare_gene_groups = resolve_rare_genes(df, cfg.rare_genes)

    print(f"There are {df.target_name.nunique()} gene types", flush=True)
    print(f"Rare gene groups used by points2regions: {rare_gene_groups}", flush=True)
    print(
        "Enabled dimensionality reductions: "
        + ", ".join(method_label(method) for method in methods),
        flush=True,
    )
    print("Running soa_points2regions_withsplit once for this run...", flush=True)

    res = points2regions_withsplit(
        xy=gene_positions,
        labels=df["target_name"].to_numpy(),
        bin_width=cfg.bin_width,
        smooth=cfg.factor,
        rare_genes=rare_gene_groups,
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

    rare_group_names = np.asarray(
        res.get("rare_group_names", np.array([], dtype=object)),
        dtype=object,
    )
    rare_group_genes = res.get("rare_group_genes", {})
    rare_group_genes_present = res.get("rare_group_genes_present", {})
    genes_all = np.asarray(res.get("genes_all", np.array([], dtype=object)), dtype=object)

    print(f"  Good bins: {good_bins.sum():,} / {len(good_bins):,}", flush=True)
    print(
        f"  X_all sparse: {X_all_sparse.shape}, nnz={X_all_sparse.nnz:,}",
        flush=True,
    )
    print(f"  X_rare: {X_rare.shape}, {X_rare.nbytes / 1e6:.1f} MB", flush=True)
    print(f"  Rare groups present: {list(rare_group_names)}", flush=True)

    df_run = df.copy()
    df_run["bin_id"] = back_map

    p2r_results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if cfg.save_p2r:
        p2r_results = run_p2r_clustering(
            res["X_all"],
            good_bins,
            cfg.k_list,
        )

    common_save_dict = build_common_save_dict(
        X_rare=X_rare,
        rare_group_names=rare_group_names,
        rare_group_genes=rare_group_genes,
        rare_group_genes_present=rare_group_genes_present,
        genes_all=genes_all,
        good_bin_ids=good_bin_ids,
        pos=pos,
        back_map=back_map,
        df_run=df_run,
        good_bins=good_bins,
        cfg=cfg,
    )

    for method in methods:
        X_norm = compute_embedding(method, X_all_sparse, cfg)
        save_cache_for_method(
            method=method,
            X_norm=X_norm,
            common_save_dict=common_save_dict,
            p2r_results=p2r_results,
            cfg=cfg,
            config=config,
            rare_gene_groups=rare_gene_groups,
            rare_group_names=rare_group_names,
            rare_group_genes_present=rare_group_genes_present,
            n_transcripts=len(df_run),
            n_genes=df_run["target_name"].nunique(),
            n_good_bins=len(good_bin_ids),
            force=force,
        )
        del X_norm
        gc.collect()

    del res, X_all_sparse, X_rare, common_save_dict, p2r_results
    del df_run, df_raw, df
    gc.collect()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_preprocess(config, force=args.force)


if __name__ == "__main__":
    main()