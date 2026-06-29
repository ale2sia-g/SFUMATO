#!/usr/bin/env python3
"""GPU BGM clustering on preprocessed SFUMATO embeddings."""

from __future__ import annotations

import argparse
import gc
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA

from bgm_pure_torch_lb import BayesianGaussianMixtureTorch
from utils_bgm import (
    BGMConfig,
    bgm_methods_from_config,
    bgm_stem_for,
    cache_file_for,
    centroids_to_hsv_rgb,
    color_mix_top2_log,
    combined_merge_distance,
    csv_to_h5ad,
    get_p2r_from_cache,
    has_rare_groups,
    image_dir_for,
    load_json_cache_field,
    load_rare_group_names,
    merge_centroids_by_groups,
    merge_probabilities_from_labels,
    plot_final_dendrogram_with_colors,
    plot_oversampled_dendrogram,
    plot_rare_group_heatmaps,
    rare_group_soft_means,
    result_dir_for,
    rgb01_to_hex,
    save_json,
    soft_cluster_centroids,
    top2_from_proba,
)
from utils_config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to JSON/YAML config.")
    return parser.parse_args()


def load_cache_for_method(cfg: BGMConfig, method: str):
    cache_file = cache_file_for(cfg, method)
    if not cache_file.exists():
        raise FileNotFoundError(f"Cache not found: {cache_file}. Run preprocess first.")

    print(f"Loading {method.upper()} cache from {cache_file}...", flush=True)
    data = np.load(cache_file, allow_pickle=True)

    rare_group_names = load_rare_group_names(data)
    rare_group_genes = load_json_cache_field(data, "rare_group_genes_json")
    rare_group_genes_present = load_json_cache_field(
        data,
        "rare_group_genes_present_json",
    )

    payload = {
        "cache_file": cache_file,
        "data": data,
        "X_norm": data["X_norm"],
        "X_rare": data["X_rare"],
        "rare_group_names": rare_group_names,
        "rare_group_genes": rare_group_genes,
        "rare_group_genes_present": rare_group_genes_present,
        "good_bin_ids": data["good_bin_ids"],
        "pos": data["pos"],
        "back_map": data["back_map"],
        "gene_x": data["gene_x"],
        "gene_y": data["gene_y"],
        "gene_name": data["gene_name"],
        "gene_bin_id": data["gene_bin_id"],
    }

    print(f"  X_norm: {payload['X_norm'].shape}", flush=True)
    print(f"  X_rare: {payload['X_rare'].shape}", flush=True)
    print(f"  Rare groups: {list(rare_group_names)}", flush=True)

    return payload


def save_merge_distances(
    outdir,
    stem: str,
    d_comb: np.ndarray,
    d_all: np.ndarray,
    d_rare: np.ndarray | None,
) -> None:
    """
    Save condensed pairwise merge distances between oversampled BGM components.

    For K_bgm components, each row is one pair of oversampled components.
    """
    n_pairs = len(d_all)
    if n_pairs == 0:
        return

    n_components = int((1 + np.sqrt(1 + 8 * n_pairs)) / 2)
    rows = []
    idx = 0
    for i in range(n_components - 1):
        for j in range(i + 1, n_components):
            rows.append(
                {
                    "component_i": i,
                    "component_j": j,
                    "d_all": float(d_all[idx]),
                    "d_rare": float(d_rare[idx]) if d_rare is not None else np.nan,
                    "d_comb": float(d_comb[idx]),
                }
            )
            idx += 1

    path = outdir / f"{stem}_oversampled_merge_distances.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Saved {path}", flush=True)


def run_bgm_for_method(config: dict, cfg: BGMConfig, method: str) -> None:
    payload = load_cache_for_method(cfg, method)

    cache_file = payload["cache_file"]
    data = payload["data"]
    X_norm = payload["X_norm"]
    X_rare = payload["X_rare"]
    rare_group_names = payload["rare_group_names"]
    rare_group_genes = payload["rare_group_genes"]
    rare_group_genes_present = payload["rare_group_genes_present"]
    good_bin_ids = payload["good_bin_ids"]
    pos = payload["pos"]
    back_map = payload["back_map"]
    gene_x = payload["gene_x"]
    gene_y = payload["gene_y"]
    gene_name = payload["gene_name"]
    gene_bin_id = payload["gene_bin_id"]

    has_rare = has_rare_groups(X_rare, rare_group_names)

    df_run = pd.DataFrame(
        {
            "x": gene_x,
            "y": gene_y,
            "target_name": gene_name,
            "bin_id": gene_bin_id,
        }
    )

    for k in cfg.k_list:
        k = int(k)
        k_bgm = int(np.ceil(cfg.bgm_oversample * k))
        oversample = cfg.bgm_oversample > 1.0
        stem = bgm_stem_for(cfg, k, method)
        outdir = result_dir_for(cfg, k, method)
        image_dir = image_dir_for(cfg, k, method)
        outdir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)

        outfile_h5ad = outdir / f"{stem}_FULL.h5ad"
        outfile_bins = outdir / f"{stem}_binlevel_FULL.csv"
        outfile_proba = outdir / f"{stem}_transcripts_proba_FULL.csv"

        print(f"\n{'=' * 60}", flush=True)
        print(f"Reduction method={method.upper()}", flush=True)
        print(f"K={k}, K_bgm={k_bgm}, oversample={oversample}", flush=True)
        print(f"beta_rare={cfg.beta_rare}", flush=True)
        print(f"Rare groups available={has_rare}", flush=True)
        print(f"Output directory: {outdir}", flush=True)

        has_p2r = cfg.save_p2r
        if has_p2r:
            cluster_p2r_bins, p2r_color_bins = get_p2r_from_cache(data, k)

        print(f"BGM ({k_bgm} components, {X_norm.shape[1]} dims)...", flush=True)
        bgm = BayesianGaussianMixtureTorch(
            n_components=k_bgm,
            n_init=1,
            init_params="k-means++",
            covariance_type="diag",
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=cfg.bgm_weight_prior,
            random_state=cfg.seed,
            max_iter=1000,
            verbose=1,
            verbose_interval=10,
            tol=1e-2,
            batch_size=None,
            device="cuda",
        )
        bgm.fit(X_norm)
        proba = bgm.predict_proba(X_norm)

        centroids_soft, masses = soft_cluster_centroids(X_norm, proba)
        centroids_for_merge = centroids_soft

        if has_rare:
            centroids_rare_soft, _ = soft_cluster_centroids(X_rare, proba)
            centroids_rare_for_merge = centroids_rare_soft
        else:
            centroids_rare_soft = None
            centroids_rare_for_merge = None

        print("  Using soft centroids", flush=True)

        Z_link = None
        d_all = None
        d_rare = None
        d_comb = None

        if oversample:
            print(f"Merging {proba.shape[1]} -> {k} clusters...", flush=True)

            d_comb, d_all, d_rare = combined_merge_distance(
                centroids_all=centroids_for_merge,
                centroids_rare=centroids_rare_for_merge,
                beta_rare=cfg.beta_rare,
            )

            if d_rare is None:
                print("  Rare groups not used in merge distance.", flush=True)
            else:
                print("  Rare groups used in merge distance.", flush=True)

            save_merge_distances(
                outdir=outdir,
                stem=stem,
                d_comb=d_comb,
                d_all=d_all,
                d_rare=d_rare,
            )

            Z_link = linkage(d_comb, method=cfg.merge_method)

            plot_oversampled_dendrogram(
                Z_link,
                target_k=k,
                path=image_dir / f"{stem}_oversampled_dendrogram_cut.png",
                title=f"{stem}: oversampled BGM dendrogram cut to K={k}",
            )

            merge_labels = fcluster(Z_link, t=k, criterion="maxclust")
            proba, groups = merge_probabilities_from_labels(proba, merge_labels)

            centroids_for_merge, masses = merge_centroids_by_groups(
                centroids_for_merge,
                masses,
                groups,
            )

            if centroids_rare_for_merge is not None:
                centroids_rare_for_merge, _ = merge_centroids_by_groups(
                    centroids_rare_for_merge,
                    np.ones(centroids_rare_for_merge.shape[0], dtype=float),
                    groups,
                )

            print(f"  Final clusters: {proba.shape[1]}", flush=True)

        rare_scores = rare_group_soft_means(
            X_rare=X_rare,
            proba=proba,
            rare_group_names=rare_group_names,
        )
        plot_rare_group_heatmaps(
            rare_scores=rare_scores,
            outdir=image_dir,
            stem=stem,
        )

        print("PCA on final cluster centroids for colors...", flush=True)
        pca_color = PCA(n_components=3, random_state=cfg.seed)
        pca_color.fit(centroids_for_merge)

        X_pca_color = pca_color.transform(X_norm)
        means_3d = pca_color.transform(centroids_for_merge)
        print(
            f"  Explained variance: {pca_color.explained_variance_ratio_.sum():.3f}",
            flush=True,
        )

        cluster, second_cluster, p1, p2 = top2_from_proba(proba)
        rgb, hue, h_shifted, h_new = centroids_to_hsv_rgb(
            means_3d,
            use_hist_equalization=cfg.use_hist_equalization,
            return_hues=True,
        )
        centroid_colors = [rgb[i] for i in range(rgb.shape[0])]

        plot_final_dendrogram_with_colors(
            centroids_for_merge,
            centroid_colors,
            image_dir / f"{stem}_final_dendrogram_colors.png",
            title=f"{stem}: final cluster dendrogram with assigned colors",
            method=cfg.merge_method,
        )

        df_bins = pd.DataFrame(
            {
                "bin_id": good_bin_ids,
                "x": pos[:, 0],
                "y": pos[:, 1],
                "cluster": cluster,
                "second_cluster": second_cluster,
                "p1": p1,
                "p2": p2,
                "compl_p1": 1 - p1,
                "PC1": X_pca_color[:, 0],
                "PC2": X_pca_color[:, 1],
                "PC3": X_pca_color[:, 2],
            }
        )
        df_bins["color_hard_hsv"] = df_bins["cluster"].apply(
            lambda c: rgb01_to_hex(centroid_colors[c])
        )
        df_bins["color_log_hsv"] = df_bins.apply(
            lambda row: color_mix_top2_log(
                centroid_colors[row["cluster"]],
                centroid_colors[row["second_cluster"]],
                row["p1"],
                row["p2"],
                alpha=50.0,
            ),
            axis=1,
        )

        if has_p2r:
            df_bins["color_p2r"] = p2r_color_bins[good_bin_ids]
            df_bins["cluster_p2r"] = cluster_p2r_bins[good_bin_ids]

        df_mappedback = df_run[["x", "y", "target_name", "bin_id"]].copy()
        df_mappedback = df_mappedback.merge(
            df_bins[
                [
                    "bin_id",
                    "cluster",
                    "color_hard_hsv",
                    "color_log_hsv",
                    "compl_p1",
                    "PC1",
                    "PC2",
                    "PC3",
                ]
            ],
            on="bin_id",
            how="left",
        )
        df_mappedback["cluster"] = (
            df_mappedback["cluster"].fillna(-1).astype(int).astype("category")
        )

        if has_p2r:
            cluster_p2r_transcripts = cluster_p2r_bins[back_map]
            df_mappedback["cluster_p2r"] = cluster_p2r_transcripts
            df_mappedback["cluster_p2r"] = df_mappedback["cluster_p2r"].astype("category")
            df_mappedback["color_p2r"] = p2r_color_bins[back_map]

        df_colors = pd.DataFrame(
            {
                "cluster": np.arange(len(hue)),
                "hue_raw": hue,
                "hue_rotated": h_shifted,
                "hue_equalized": h_new,
                "color_hex": [rgb01_to_hex(c) for c in rgb],
            }
        )

        hues_file = outdir / f"hues_{stem}.csv"
        df_colors.to_csv(hues_file, index=False)
        print(f"  Saved {hues_file}", flush=True)

        adata_output = csv_to_h5ad(df_mappedback)
        adata_output.write_h5ad(outfile_h5ad)
        print(f"  Saved {outfile_h5ad}", flush=True)

        df_bins.to_csv(outfile_bins, index=False)
        print(f"  Saved {outfile_bins}", flush=True)

        weights_file = outdir / f"weights_{stem}.txt"
        with weights_file.open("w", encoding="utf-8") as f:
            f.write(f"Reduction method={method.upper()}\n")
            f.write(f"K={k}, K_bgm={k_bgm}\n")
            f.write(f"beta_rare={cfg.beta_rare}\n")
            f.write(f"rare_groups_available={has_rare}\n")
            f.write(f"rare_groups_used_in_merge={d_rare is not None}\n")
            f.write(f"rare_group_names={list(rare_group_names)}\n")
            f.write(f"rare_group_genes={rare_group_genes}\n")
            f.write(f"rare_group_genes_present={rare_group_genes_present}\n")
            f.write("\nBGM weights:\n")
            for i, weight in enumerate(bgm.weights_):
                f.write(f"Component {i}: {weight}\n")
            f.write("\nCluster masses:\n")
            for j, mass in enumerate(masses):
                f.write(f"Cluster {j}: {mass}\n")
        print(f"  Saved {weights_file}", flush=True)

        if cfg.save_transcript_proba:
            print("Saving transcript-level probability CSV...", flush=True)
            n_clusters_final = proba.shape[1]
            transcript_bin_ids = df_run["bin_id"].to_numpy()
            n_bins_total = (
                int(data["n_bins_total"][0])
                if "n_bins_total" in data
                else int(transcript_bin_ids.max()) + 1
            )
            bin_id_to_row = np.full(n_bins_total, -1, dtype=np.int32)
            bin_id_to_row[good_bin_ids] = np.arange(len(good_bin_ids), dtype=np.int32)
            transcript_proba_rows = bin_id_to_row[transcript_bin_ids]

            proba_transcripts = np.zeros(
                (len(df_run), n_clusters_final),
                dtype=np.float32,
            )
            valid_mask = transcript_proba_rows >= 0
            proba_transcripts[valid_mask] = proba[
                transcript_proba_rows[valid_mask]
            ].astype(np.float32)

            df_proba = pd.DataFrame(
                {
                    "x": gene_x.astype(np.int32),
                    "y": gene_y.astype(np.int32),
                    "target_name": gene_name,
                    "bin_id": transcript_bin_ids.astype(np.int32),
                    "color_hard_hsv": df_mappedback["color_hard_hsv"].to_numpy(),
                    "color_log_hsv": df_mappedback["color_log_hsv"].to_numpy(),
                }
            )
            for cluster_idx in range(n_clusters_final):
                df_proba[f"p{cluster_idx + 1}"] = proba_transcripts[:, cluster_idx]

            df_proba.to_csv(outfile_proba, index=False)
            print(
                f"  Saved {outfile_proba} - {len(df_proba):,} transcripts, "
                f"{n_clusters_final} clusters",
                flush=True,
            )
            del df_proba, proba_transcripts, bin_id_to_row, transcript_proba_rows
            gc.collect()
        else:
            print("  Skipped transcript-level probability CSV.", flush=True)

        save_json(
            outdir / f"{stem}_bgm_config.json",
            {
                "run_name": cfg.run_name,
                "reduction_method": method.upper(),
                "K": k,
                "K_bgm": k_bgm,
                "cache_file": str(cache_file),
                "output_directory": str(outdir),
                "bgm": config.get("bgm", {}),
                "preprocess": config.get("preprocess", {}),
                "save_transcript_proba": cfg.save_transcript_proba,
                "beta_rare": cfg.beta_rare,
                "rare_groups_available": has_rare,
                "rare_groups_used_in_merge": d_rare is not None,
                "rare_group_names": [str(x) for x in rare_group_names],
                "rare_group_genes": rare_group_genes,
                "rare_group_genes_present": rare_group_genes_present,
            },
        )

        del bgm, proba, centroids_soft
        del centroids_for_merge, masses
        del pca_color, X_pca_color, means_3d
        del cluster, second_cluster, p1, p2, rgb, hue, h_shifted, h_new
        del centroid_colors, df_bins, df_mappedback, df_colors, adata_output
        del rare_scores
        if centroids_rare_soft is not None:
            del centroids_rare_soft
        if centroids_rare_for_merge is not None:
            del centroids_rare_for_merge
        if Z_link is not None:
            del Z_link
        if d_all is not None:
            del d_all
        if d_rare is not None:
            del d_rare
        if d_comb is not None:
            del d_comb
        gc.collect()
        print(f"  Done {method.upper()} K={k}", flush=True)

    data.close()
    del data, X_norm, X_rare, rare_group_names
    del rare_group_genes, rare_group_genes_present
    del good_bin_ids, pos, back_map
    del gene_x, gene_y, gene_name, gene_bin_id, df_run
    gc.collect()


def run_bgm(config: dict) -> None:
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        raise SystemExit("CUDA is required for BGM.")

    cfg = BGMConfig.from_dict(config)
    if cfg.merge_metric != "cosine":
        raise ValueError("Only merge_metric='cosine' is currently implemented.")

    methods = bgm_methods_from_config(cfg)
    print(f"Selected BGM reduction methods: {methods}", flush=True)

    for method in methods:
        run_bgm_for_method(config, cfg, method)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_bgm(config)


if __name__ == "__main__":
    main()