# SFUMATO

## A Bayesian probabilistic clustering and visualisation method for uncertainty-aware spatial transcriptomics

<p align="center">
  <img src="assets/monalisa.png" width="150" alt="MonaLisa logo">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/uu.png" width="200" alt="Uppsala University logo">
</p>

<p align="center">
  <strong>Developed at Uppsala University in the Wählby group</strong>
</p>

---

SFUMATO is a Bayesian probabilistic clustering and visualisation pipeline for spatial transcriptomics.
It uses spatial binning, dimensionality reduction, and Bayesian Gaussian mixture modelling to identify spatial niches while preserving uncertainty information and enabling interpretable colour-based visualisation.

The pipeline is config-driven: the same config file is used for CPU preprocessing, GPU BGM clustering, local runs, optional UMAP/embedding plots, and Alvis SLURM jobs.

## Main Config

Example:

```bash
configs/mousexeniumniche.json
```

The config controls:

- dataset name and input CSV
- coordinate/gene column names
- binning and smoothing parameters
- dimensionality reduction method, SVD and/or PCA
- BGM cluster numbers
- optional P2R clustering
- optional rare marker groups
- output directories
- optional UMAP settings

Example structure:

```json
{
  "run_name": "mousexeniumniche",
  "input_csv": "xenium_mouse.csv",
  "columns": {
    "x": "x",
    "y": "y",
    "gene": "target_name",
    "num_id": "num_id"
  },
  "preprocess": {
    "k_list": [15, 20, 25, 30],
    "bgm_oversample": 1.5,
    "min_genes_per_bin": 120,
    "bin_width": 40,
    "factor": 8,
    "seed": 8,
    "use_svd": true,
    "use_pca": true,
    "comp": 10,
    "save_p2r": true,
    "alpha": 1.0,
    "rare_genes": null,
    "cache_dir": "preprocess_cache_mousexeniumniche"
  },
  "bgm": {
    "outroot": "results_mousexeniumniche",
    "save_transcript_proba": false,
    "beta_rare": 0.0,
    "merge_method": "complete",
    "merge_metric": "cosine",
    "bgm_weight_prior": 1,
    "use_hist_equalization": true
  },
  "umap": {
    "run": false,
    "max_points": 200000,
    "umap_components": 3,
    "n_neighbors": 30,
    "min_dist": 0.15,
    "point_size": 0.35
  }
}
```

## SVD and PCA

Dimensionality reduction is controlled by:

```json
"use_svd": true,
"use_pca": true,
"comp": 10
```

If both `use_svd` and `use_pca` are `true`, preprocessing runs points2regions once and then writes two separate caches:

```text
preprocess_cache_mousexeniumniche_svd/
preprocess_cache_mousexeniumniche_pca/
```

BGM then runs separately on each enabled representation and writes method-specific results:

```text
results_mousexeniumniche_svd/
results_mousexeniumniche_pca/
```

For large or very fine-bin datasets, PCA may require much more RAM because it densifies the sparse matrix. SVD is the safer default for large sparse spatial transcriptomics inputs.

## Rare Marker Groups

Rare marker genes can optionally be provided as named marker groups:

```json
"rare_genes": {
  "celltype_A": ["Gene1"],
  "celltype_B": ["Gene2", "Gene3"],
  "celltype_C": ["Gene4", "Gene5", "Gene6"]
}
```

If `rare_genes` is `null` or `{}`, no fallback gene is selected. The rare-gene matrix is saved as an empty matrix with shape:

```text
n_bins x 0
```

This preserves the ordinary SFUMATO flow exactly when rare genes are not used.

When rare marker groups are provided, points2regions returns one score per group:

```text
X_rare: n_bins x n_rare_groups
```

These rare-group scores do not drive the initial BGM fit. The BGM is still fitted on the SVD/PCA embedding stored in `X_norm`.

Rare marker groups can influence only the hierarchical merge from oversampled `K_bgm` components to the final target `K`, controlled by:

```json
"beta_rare": 0.0
```

Use:

```json
"beta_rare": 0.0
```

to ignore rare markers during merging.

Use a positive value, for example:

```json
"beta_rare": 0.5
```

to add rare-marker-group distances to the merge distance.

The merge distance is:

```text
d_comb = d_all + beta_rare * d_rare
```

where:

- `d_all` is the distance between BGM component centroids in the SVD/PCA embedding.
- `d_rare` is computed pairwise between oversampled BGM components: for each pair of components, SFUMATO computes the distance for every rare marker group and keeps the largest one.

The max across groups is used so that a strong difference in any one rare marker group is not diluted by unrelated groups.

## Rare Marker Outputs

When rare marker groups are available, BGM writes diagnostic outputs per run:

```text
*_rare_group_soft_means.csv
*_rare_group_soft_means_row_zscore.csv
images/*_rare_group_soft_means_heatmap.png
images/*_rare_group_soft_means_row_zscore_heatmap.png
```

Rows are rare marker groups and columns are final SFUMATO clusters.

The raw heatmap shows the soft mean rare-group score per cluster.
The row-z-scored heatmap shows relative enrichment of each rare marker group across clusters.

These heatmaps are diagnostics. They help interpret whether rare marker groups are actually represented in the final clusters.

## Local / Workstation

Run preprocessing and BGM in one command:

```bash
python run_sfumato.py --config configs/mousexeniumniche.json
```

Or run the two steps separately:

```bash
python preprocess.py --config configs/mousexeniumniche.json
python run_bgm_gpu.py --config configs/mousexeniumniche.json
```

To overwrite existing preprocessing caches:

```bash
python preprocess.py --config configs/mousexeniumniche.json --force
```

or:

```bash
python run_sfumato.py --config configs/mousexeniumniche.json --force-preprocess
```

## Alvis / SLURM

Submit preprocessing and BGM as separate jobs with dependency:

```bash
bash sbatch/submit_pipeline.sh
```

The preprocessing job uses:

```bash
sbatch/preprocess.sbatch
```

The GPU job uses:

```bash
sbatch/run_bgm_gpu.sbatch
```

For CPU-only preprocessing on Alvis, memory is allocated proportionally to requested CPU cores. For high-memory preprocessing, use the appropriate NOGPU memory feature and CPU allocation for the target node type.

## Outputs

Example preprocessing cache when SVD is enabled:

```text
preprocess_cache_mousexeniumniche_svd/
  mousexeniumniche_BIN40_F8_SVD10.npz
  mousexeniumniche_BIN40_F8_SVD10_preprocess_config.json
```

Example preprocessing cache when PCA is enabled:

```text
preprocess_cache_mousexeniumniche_pca/
  mousexeniumniche_BIN40_F8_PCA10.npz
  mousexeniumniche_BIN40_F8_PCA10_preprocess_config.json
```

BGM outputs are grouped by `K` and dimensionality-reduction method:

```text
results_mousexeniumniche_svd/
  K30/
    SVD10/
      *_FULL.h5ad
      *_binlevel_FULL.csv
      hues_*.csv
      weights_*.txt
      *_oversampled_merge_distances.csv
      *_bgm_config.json
      images/
        *_oversampled_dendrogram_cut.png
        *_final_dendrogram_colors.png
        *_rare_group_soft_means_heatmap.png
        *_rare_group_soft_means_row_zscore_heatmap.png
```

The transcript-level probability CSV is skipped by default. Enable it with:

```json
"save_transcript_proba": true
```

## Optional UMAP Figures

UMAP post-processing:

```bash
python plot_embedding_umap.py --config configs/mousexeniumniche.json
```

For one K only:

```bash
python plot_embedding_umap.py --config configs/mousexeniumniche.json --k 30
```

## Optional Embedding-Dimension Figures

Plot the first embedding dimensions directly:

```bash
python plot_embedding_dims.py --config configs/mousexeniumniche.json --k 30
```

This is useful for checking whether the low-dimensional representation already contains clear structure before UMAP.

## Hue Equalization Figure

Hue equalization figure from a `hues_*.csv` file:

```bash
python plot_hue_equalization.py \
  --hues-csv results_mousexeniumniche_svd/K30/SVD10/hues_mousexeniumniche_BGM45to30_BIN40_F8_SVD10.csv \
  --output results_mousexeniumniche_svd/K30/SVD10/images/mousexeniumniche_K30_SVD10_hue_equalization.png
```

## Notes

- `X_norm` is the embedding used by BGM.
- `X_rare` is optional and contains rare marker-group scores.
- Without rare marker groups, the pipeline follows the ordinary SFUMATO path.
- Rare marker groups can preserve rare-marker-enriched oversampled BGM components during hierarchical merging, but they cannot recover a rare component that was never separated by the initial BGM fit.
- SFUMATO colours are assigned from PCA of final cluster centroids in embedding space; rare marker groups affect colours only indirectly if they change the final merge.