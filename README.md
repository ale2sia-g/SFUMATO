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
It uses spatial binning, sparse dimensionality reduction, and Bayesian Gaussian mixture modelling to identify spatial niches while preserving uncertainty information and enabling interpretable colour-based visualisation.

The pipeline is config-driven: the same config file is used for CPU preprocessing, GPU BGM clustering, local runs, optional UMAP/embedding plots, and Alvis SLURM jobs.

## Repository Structure

Recommended structure:

```text
SFUMATO/
  bgm/
    __init__.py
    bgm_pure_torch_lb.py
    bgm_torch_ops.py
    memory_utils.py

  configs/
    mousexeniumniche.json

  sbatch/
    preprocess.sbatch
    run_bgm_gpu.sbatch
    submit_pipeline.sh

  transcripts_files/
    xenium_mouse.csv

  notebooks/
  assets/

  points2regions.py
  preprocess.py
  run_bgm_gpu.py
  run_sfumato.py
  utils_bgm.py
  utils_config.py
  utils_preprocess.py
```

Transcript-level input files should be placed in:

```text
transcripts_files/
```

For example:

```json
"input_csv": "transcripts_files/xenium_mouse.csv"
```

## Main Config

Example:

```bash
configs/mousexeniumniche.json
```

The config controls:

- dataset name and input CSV
- coordinate/gene column names
- binning and smoothing parameters
- sparse SVD dimensionality reduction
- BGM cluster resolutions
- optional P2R clustering
- output directories
- hierarchical colour assignment
- optional UMAP settings

Example structure:

```json
{
  "run_name": "mousexeniumniche",
  "input_csv": "transcripts_files/xenium_mouse.csv",
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
    "comp": 10,
    "save_p2r": true,
    "alpha": 1.0,
    "cache_dir": "preprocess_cache_mousexeniumniche"
  },
  "bgm": {
    "outroot": "results_mousexeniumniche",
    "save_transcript_proba": false,
    "merge_method": "complete",
    "merge_metric": "cosine",
    "bgm_weight_prior": 1
  },
  "color": {
    "colormap": "gist_ncar",
    "colormap_start": 0.08,
    "colormap_end": 0.92,
    "dendrogram_branch_linewidth": 3.0
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

## Preprocessing

Preprocessing runs a Points2Regions-style feature extraction step followed by sparse TruncatedSVD.

The main preprocessing parameters are:

```json
"bin_width": 40,
"factor": 8,
"min_genes_per_bin": 120,
"comp": 10
```

The output cache contains:

```text
X_norm        sparse-SVD embedding used by BGM
good_bin_ids  bin ids retained after filtering
pos           spatial bin coordinates
back_map      transcript-to-bin mapping
gene_x/y      transcript coordinates
gene_name     transcript gene labels
gene_bin_id   transcript bin ids
```

Sparse SVD is used instead of dense PCA because spatial transcriptomics bin-by-gene matrices are typically sparse and can be too large to densify.

## Shared Multi-Resolution BGM

SFUMATO fits one oversampled Bayesian Gaussian mixture model using the largest requested K:

```text
K_max = max(k_list)
K_bgm = ceil(bgm_oversample * K_max)
```

For example, if:

```json
"k_list": [15, 20, 25, 30],
"bgm_oversample": 1.5
```

then:

```text
K_max = 30
K_bgm = 45
```

The GPU step then:

1. fits one BGM with `K_bgm` components;
2. computes soft centroids for the oversampled components;
3. builds one shared hierarchical dendrogram of the oversampled components;
4. cuts the same dendrogram at each requested K;
5. saves one set of outputs for each K.

This ensures that different K resolutions are derived from the same model and the same dendrogram.

## Hierarchical Colour Assignment

SFUMATO assigns cluster colours from the shared dendrogram ordering.

The default colour ramp is:

```json
"color": {
  "colormap": "gist_ncar",
  "colormap_start": 0.08,
  "colormap_end": 0.92
}
```

This uses a trimmed `gist_ncar` colormap to avoid the darkest and most extreme endpoints while preserving a broad range of distinguishable colours.

For each K:

1. the shared dendrogram is cut at K;
2. each final cluster receives a colour according to the span of its oversampled BGM components along the optimal dendrogram leaf ordering;
3. finer K values inherit related colours from the same broader dendrogram branches.

This gives colour consistency across resolutions: a broad branch at low K is split into related colour shades at higher K.

The user can choose another Matplotlib continuous colormap in the config. For example, colourblind-friendly alternatives can be tested with:

```json
"color": {
  "colormap": "viridis",
  "colormap_start": 0.0,
  "colormap_end": 1.0
}
```

or:

```json
"color": {
  "colormap": "cividis",
  "colormap_start": 0.0,
  "colormap_end": 1.0
}
```

Note that different colormaps have different perceptual properties. The default `gist_ncar` setting prioritises visual separability across many clusters.

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

To overwrite an existing preprocessing cache:

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

Example preprocessing cache:

```text
preprocess_cache_mousexeniumniche/
  mousexeniumniche_BIN40_F8_SVD10.npz
  mousexeniumniche_BIN40_F8_SVD10_preprocess_config.json
```

Shared BGM outputs:

```text
results_mousexeniumniche/
  shared_BGM/
    SVD10/
      mousexeniumniche_BGM45toMAX30_BIN40_F8_SVD10_cluster_color_table_allK.csv
      mousexeniumniche_BGM45toMAX30_BIN40_F8_SVD10_oversampled_merge_distances.csv
      mousexeniumniche_BGM45toMAX30_BIN40_F8_SVD10_shared_bgm_config.json
      weights_mousexeniumniche_BGM45toMAX30_BIN40_F8_SVD10.txt
      images/
        mousexeniumniche_BGM45toMAX30_BIN40_F8_SVD10_multiresolution_dendrogram_allK.png
```

K-specific outputs:

```text
results_mousexeniumniche/
  K30/
    SVD10/
      *_FULL.h5ad
      *_binlevel_FULL.csv
      colors_*.csv
      weights_*.txt
      *_bgm_config.json
      images/
        *_cut_dendrogram.png
```

The transcript-level probability CSV is skipped by default. Enable it with:

```json
"save_transcript_proba": true
```

## Dendrogram Figures

SFUMATO saves two types of dendrogram figures.

The shared multiresolution dendrogram:

```text
*_multiresolution_dendrogram_allK.png
```

This figure shows:

- the shared oversampled-component dendrogram;
- branches coloured across resolutions;
- one dashed horizontal cut line per K;
- aligned colour strips showing cluster colours at each K.

The K-specific dendrogram:

```text
*_cut_dendrogram.png
```

This figure shows:

- the same shared dendrogram;
- branches coloured according to the selected K;
- one dashed cut line;
- one colour strip for that K.

These figures are useful for checking that colour assignment is consistent across resolutions.

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

This is useful for checking whether the SVD representation already contains clear structure before UMAP.

## Notes

- `X_norm` is the sparse-SVD embedding used by BGM.
- BGM is fitted once using `K_bgm = ceil(bgm_oversample * max(k_list))`.
- All requested K values are obtained by cutting the same shared dendrogram.
- Cluster colours are assigned from the shared dendrogram ordering, not from a separate PCA of final centroids.
- Colour differences should be interpreted as following the shared hierarchical ordering, not as exact metric distances in expression space.