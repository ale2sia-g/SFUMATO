#!/usr/bin/env python3
"""GPU BGM clustering on preprocessed SFUMATO SVD embeddings.

This version fits one oversampled BGM using max(K_LIST), builds one shared
component dendrogram, and obtains all requested K values by cutting the same
dendrogram. Cluster colors are assigned from the shared dendrogram ordering.
"""

from __future__ import annotations

import argparse
import gc
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import linkage

from bgm.bgm_pure_torch_lb import BayesianGaussianMixtureTorch
from utils_bgm import (
    BGMConfig,
    bgm_stem_for,
    cache_file_for,
    cluster_color_table,
    color_mix_top2_log,
    colors_for_cut,
    csv_to_h5ad,
    get_continuous_colormap_func,
    get_p2r_from_cache,
    image_dir_for,
    leaf_rank_from_linkage,
    merge_centroids_by_groups,
    merge_probabilities_from_labels,
    normalize_condensed,
    plot_cut_dendrogram,
    plot_multiresolution_dendrogram,
    result_dir_for,
    safe_cosine_pdist,
    save_json,
    shared_bgm_stem_for,
    shared_image_dir_for,
    shared_result_dir_for,
    soft_cluster_centroids,
    top2_from_proba,
)
from utils_config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to JSON/YAML config.")
    return parser.parse_args()


def load_cache(cfg: BGMConfig) -> dict:
    cache_file = cache_file_for(cfg)
    if not cache_file.exists():
        raise FileNotFoundError(f"Cache not found: {cache_file}. Run preprocess first.")

    print(f"Loading cache from {cache_file}...", flush=True)
    data = np.load(cache_file, allow_pickle=True)

    payload = {
        "cache_file": cache_file,
        "data": data,
        "X_norm": data["X_norm"],
        "good_bin_ids": data["good_bin_ids"],
        "pos": data["pos"],
        "back_map": data["back_map"],
        "gene_x": data["gene_x"],
        "gene_y": data["gene_y"],
        "gene_name": data["gene_name"],
        "gene_bin_id": data["gene_bin_id"],
    }

    print(f"  X_norm: {payload['X_norm'].shape}", flush=True)
    return payload


def save_oversampled_merge_distances(
    outdir,
    stem: str,
    d_all: np.ndarray,
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
                }
            )
            idx += 1

    path = outdir / f"{stem}_oversampled_merge_distances.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Saved {path}", flush=True)


def first_svd_dims(X_norm: np.ndarray, n_dims: int = 3) -> np.ndarray:
    """
    Return first SVD dimensions padded to n_dims.

    These are saved for downstream inspection. They are not used for color
    assignment.
    """
    out = np.zeros((X_norm.shape[0], n_dims), dtype=np.float32)
    n = min(n_dims, X_norm.shape[1])
    out[:, :n] = X_norm[:, :n]
    return out


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

    payload = load_cache(cfg)

    cache_file = payload["cache_file"]
    data = payload["data"]
    X_norm = payload["X_norm"]
    good_bin_ids = payload["good_bin_ids"]
    pos = payload["pos"]
    back_map = payload["back_map"]
    gene_x = payload["gene_x"]
    gene_y = payload["gene_y"]
    gene_name = payload["gene_name"]
    gene_bin_id = payload["gene_bin_id"]

    df_run = pd.DataFrame(
        {
            "x": gene_x,
            "y": gene_y,
            "target_name": gene_name,
            "bin_id": gene_bin_id,
        }
    )

    k_list = sorted([int(k) for k in cfg.k_list])
    k_max = max(k_list)
    k_bgm = int(np.ceil(cfg.bgm_oversample * k_max))
    oversample = cfg.bgm_oversample > 1.0

    shared_stem = shared_bgm_stem_for(cfg, k_bgm, k_max)
    shared_outdir = shared_result_dir_for(cfg)
    shared_image_dir = shared_image_dir_for(cfg)
    shared_outdir.mkdir(parents=True, exist_ok=True)
    shared_image_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}", flush=True)
    print("Shared BGM run", flush=True)
    print(f"K_LIST={k_list}", flush=True)
    print(f"K_max={k_max}", flush=True)
    print(f"bgm_oversample={cfg.bgm_oversample}", flush=True)
    print(f"K_bgm={k_bgm}", flush=True)
    print(f"oversample={oversample}", flush=True)
    print(f"Output directory: {shared_outdir}", flush=True)

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
    proba_bgm = bgm.predict_proba(X_norm)

    centroids_bgm, masses_bgm = soft_cluster_centroids(X_norm, proba_bgm)
    print("  Using soft centroids for shared dendrogram", flush=True)

    d_all = normalize_condensed(safe_cosine_pdist(centroids_bgm))
    save_oversampled_merge_distances(shared_outdir, shared_stem, d_all)

    Z_link = linkage(
        d_all,
        method=cfg.merge_method,
        optimal_ordering=True,
    )

    leaf_rank = leaf_rank_from_linkage(Z_link)
    color_func = get_continuous_colormap_func(
        cmap_name=cfg.color_colormap,
        start=cfg.color_colormap_start,
        end=cfg.color_colormap_end,
    )

    color_table = cluster_color_table(Z_link, k_list, color_func)
    color_table_file = shared_outdir / f"{shared_stem}_cluster_color_table_allK.csv"
    color_table.to_csv(color_table_file, index=False)
    print(f"  Saved {color_table_file}", flush=True)

    plot_multiresolution_dendrogram(
        Z=Z_link,
        k_list=k_list,
        color_func=color_func,
        path=shared_image_dir / f"{shared_stem}_multiresolution_dendrogram_allK.png",
        title=f"{shared_stem}: shared multiresolution dendrogram",
        branch_linewidth=cfg.dendrogram_branch_linewidth,
    )
    print("  Saved shared multiresolution dendrogram", flush=True)

    weights_file = shared_outdir / f"weights_{shared_stem}.txt"
    with weights_file.open("w", encoding="utf-8") as f:
        f.write("Shared oversampled BGM\n")
        f.write(f"K_LIST={k_list}\n")
        f.write(f"K_max={k_max}\n")
        f.write(f"K_bgm={k_bgm}\n")
        f.write(f"bgm_oversample={cfg.bgm_oversample}\n")
        f.write(f"merge_method={cfg.merge_method}\n")
        f.write(f"merge_metric={cfg.merge_metric}\n")
        f.write(f"color_colormap={cfg.color_colormap}\n")
        f.write(f"color_colormap_start={cfg.color_colormap_start}\n")
        f.write(f"color_colormap_end={cfg.color_colormap_end}\n")
        f.write("\nBGM weights:\n")
        for i, weight in enumerate(bgm.weights_):
            f.write(f"Component {i}: {weight}\n")
        f.write("\nOversampled component masses:\n")
        for j, mass in enumerate(masses_bgm):
            f.write(f"Component {j}: {mass}\n")
    print(f"  Saved {weights_file}", flush=True)

    save_json(
        shared_outdir / f"{shared_stem}_shared_bgm_config.json",
        {
            "run_name": cfg.run_name,
            "K_LIST": k_list,
            "K_max": k_max,
            "K_bgm": k_bgm,
            "cache_file": str(cache_file),
            "output_directory": str(shared_outdir),
            "bgm": config.get("bgm", {}),
            "preprocess": config.get("preprocess", {}),
            "color": config.get("color", {}),
            "save_transcript_proba": cfg.save_transcript_proba,
        },
    )

    for k in k_list:
        stem = bgm_stem_for(cfg, k, k_bgm)
        outdir = result_dir_for(cfg, k)
        image_dir = image_dir_for(cfg, k)
        outdir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)

        outfile_h5ad = outdir / f"{stem}_FULL.h5ad"
        outfile_bins = outdir / f"{stem}_binlevel_FULL.csv"
        outfile_proba = outdir / f"{stem}_transcripts_proba_FULL.csv"

        print(f"\n{'=' * 60}", flush=True)
        print(f"Cutting shared dendrogram to K={k}", flush=True)
        print(f"Output directory: {outdir}", flush=True)

        merge_labels, spans, colors_hex_by_label, colors_rgb_by_label = colors_for_cut(
            Z_link,
            k,
            leaf_rank,
            color_func,
        )

        unique_merge_labels = np.unique(merge_labels)

        proba, groups = merge_probabilities_from_labels(proba_bgm, merge_labels)

        centroids_final, masses_final = merge_centroids_by_groups(
            centroids_bgm,
            masses_bgm,
            groups,
        )

        actual_k = proba.shape[1]
        print(f"  Final clusters: {actual_k}", flush=True)

        centroid_colors = [
            colors_rgb_by_label[int(label)] for label in unique_merge_labels
        ]
        centroid_colors_hex = [
            colors_hex_by_label[int(label)] for label in unique_merge_labels
        ]

        plot_cut_dendrogram(
            Z=Z_link,
            k=k,
            color_func=color_func,
            path=image_dir / f"{stem}_cut_dendrogram.png",
            title=f"{stem}: shared dendrogram cut to K={k}",
            branch_linewidth=cfg.dendrogram_branch_linewidth,
        )
        print(f"  Saved cut dendrogram for K={k}", flush=True)

        cluster, second_cluster, p1, p2 = top2_from_proba(proba)

        alpha = 50.0
        w1 = np.log1p(alpha * p1)
        w2 = np.log1p(alpha * p2)
        mix_t = (w2 / (w1 + w2)).astype(np.float32)

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
                "mix_t": mix_t,
            }
        )

        df_bins["color_hard_hsv"] = df_bins["cluster"].apply(
            lambda c: centroid_colors_hex[int(c)]
        )
        df_bins["color_log_hsv"] = df_bins.apply(
            lambda row: color_mix_top2_log(
                centroid_colors[int(row["cluster"])],
                centroid_colors[int(row["second_cluster"])],
                row["p1"],
                row["p2"],
                alpha=50.0,
            ),
            axis=1,
        )

        if cfg.save_p2r:
            cluster_p2r_bins, p2r_color_bins = get_p2r_from_cache(data, k)
            df_bins["color_p2r"] = p2r_color_bins[good_bin_ids]
            df_bins["cluster_p2r"] = cluster_p2r_bins[good_bin_ids]

        df_mappedback = df_run[["x", "y", "target_name", "bin_id"]].copy()
        df_mappedback = df_mappedback.merge(
            df_bins[
                [
                    "bin_id",
                    "cluster",
                    "second_cluster",
                    "color_hard_hsv",
                    "color_log_hsv",
                    "compl_p1",
                    "mix_t",
                ]
            ],
            on="bin_id",
            how="left",
        )
        df_mappedback["cluster"] = (
            df_mappedback["cluster"].fillna(-1).astype(int).astype("category")
        )
        df_mappedback["second_cluster"] = (
            df_mappedback["second_cluster"]
            .fillna(-1)
            .astype(int)
            .astype("category")
        )

        if cfg.save_p2r:
            cluster_p2r_transcripts = cluster_p2r_bins[back_map]
            df_mappedback["cluster_p2r"] = cluster_p2r_transcripts
            df_mappedback["cluster_p2r"] = df_mappedback["cluster_p2r"].astype("category")
            df_mappedback["color_p2r"] = p2r_color_bins[back_map]

        df_colors = pd.DataFrame(
            {
                "cluster": np.arange(actual_k),
                "merge_label": unique_merge_labels.astype(int),
                "color_hex": centroid_colors_hex,
                "n_oversampled_components": [
                    len(spans[int(label)]["members"]) for label in unique_merge_labels
                ],
                "oversampled_components": [
                    ",".join(str(int(x)) for x in spans[int(label)]["members"])
                    for label in unique_merge_labels
                ],
            }
        )

        colors_file = outdir / f"colors_{stem}.csv"
        df_colors.to_csv(colors_file, index=False)
        print(f"  Saved {colors_file}", flush=True)

        adata_output = csv_to_h5ad(df_mappedback)
        adata_output.write_h5ad(outfile_h5ad)
        print(f"  Saved {outfile_h5ad}", flush=True)

        df_bins.to_csv(outfile_bins, index=False)
        print(f"  Saved {outfile_bins}", flush=True)

        weights_k_file = outdir / f"weights_{stem}.txt"
        with weights_k_file.open("w", encoding="utf-8") as f:
            f.write("Shared BGM cut\n")
            f.write(f"K={k}\n")
            f.write(f"actual_K={actual_k}\n")
            f.write(f"K_bgm={k_bgm}\n")
            f.write(f"bgm_oversample={cfg.bgm_oversample}\n")
            f.write(f"shared_stem={shared_stem}\n")
            f.write(f"color_colormap={cfg.color_colormap}\n")
            f.write(f"color_colormap_start={cfg.color_colormap_start}\n")
            f.write(f"color_colormap_end={cfg.color_colormap_end}\n")
            f.write("\nFinal cluster masses:\n")
            for j, mass in enumerate(masses_final):
                f.write(f"Cluster {j}: {mass}\n")
            f.write("\nCluster colors:\n")
            for j, color in enumerate(centroid_colors_hex):
                f.write(f"Cluster {j}: {color}\n")
        print(f"  Saved {weights_k_file}", flush=True)

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
                "K": k,
                "actual_K": actual_k,
                "K_bgm": k_bgm,
                "shared_stem": shared_stem,
                "cache_file": str(cache_file),
                "output_directory": str(outdir),
                "shared_output_directory": str(shared_outdir),
                "bgm": config.get("bgm", {}),
                "preprocess": config.get("preprocess", {}),
                "color": config.get("color", {}),
                "save_transcript_proba": cfg.save_transcript_proba,
            },
        )

        del proba, groups, centroids_final, masses_final
        del merge_labels, unique_merge_labels
        del centroid_colors, centroid_colors_hex
        del cluster, second_cluster, p1, p2, mix_t, w1, w2
        del df_bins, df_mappedback, df_colors, adata_output
        if cfg.save_p2r:
            del cluster_p2r_bins, p2r_color_bins
        gc.collect()
        print(f"  Done K={k}", flush=True)

    data.close()

    del data, X_norm, proba_bgm, bgm
    del centroids_bgm, masses_bgm, d_all, Z_link, color_table
    del good_bin_ids, pos, back_map
    del gene_x, gene_y, gene_name, gene_bin_id, df_run
    gc.collect()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_bgm(config)


if __name__ == "__main__":
    main()