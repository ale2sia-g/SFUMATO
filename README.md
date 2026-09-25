# SFUMATO

## Bayesian probabilistic clustering for uncertainty-aware spatial transcriptomics analysis and mapping

<p align="center">
  <img src="assets/monalisa.png" width="150" alt="MonaLisa logo">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/uu.png" width="200" alt="Uppsala University logo">
</p>

<p align="center">
  <strong>Developed at Uppsala University, Wählby group</strong>
</p>

---

SFUMATO is a Bayesian probabilistic clustering and visualisation pipeline for spatial transcriptomics.
It uses spatial binning, sparse dimensionality reduction, and Bayesian Gaussian mixture modelling to identify spatial niches while preserving uncertainty information and enabling interpretable colour-based visualisation.

The pipeline is config-driven: the same config file is used for CPU preprocessing, GPU BGM clustering, local runs, optional UMAP/embedding plots, and Alvis SLURM jobs.

## Repository Structure

```text
SFUMATO/
  bgm/
    __init__.py
    bgm_pure_torch_lb.py
    bgm_torch_ops.py
    memory_utils.py

  configs/
    mousexeniumniche.json       ← example config (mouse Xenium, niche level)
    mousexeniumcell.json
    breastxeniumniche.json
    breastxeniumcell.json
    ...

  make_masks/
    make_cluster_masks_from_binlevel_posterior.py
    make_sfumato_semantic_masks_breast.py

  notebooks/
    examples/
      sfumato_full_pipeline.ipynb         ← full pipeline walkthrough with toy data
      sfumato_bgm_custom_features.ipynb   ← run BGM on your own pre-computed features
      embedding_from_cache.ipynb
    paper_analysis/
      compare_sfumato_ficture.ipynb
      compare_sfumato_points2regions.ipynb
      compare_sfumato_xenium.ipynb
      dge_cell_level_45_clusters_dendrogram_aligned_plot_only.ipynb

  plots_umap/
    plot_embedding_dims.py
    plot_embedding_umap.py

  sbatch/
    preprocess.sbatch
    run_bgm_gpu.sbatch
    submit_pipeline.sh

  transcripts_files/            ← place your input CSV here (not tracked by git)

  points2regions.py
  preprocess.py
  run_bgm_gpu.py
  run_sfumato.py
  utils_bgm.py
  utils_config.py
  utils_preprocess.py
  requirements.txt
```

Transcript-level input files should be placed in `transcripts_files/`. For example:

```json
"input_csv": "transcripts_files/xenium_mouse.csv"
```

## Installation

```bash
pip install -r requirements.txt
```

PyTorch must be installed with CUDA support. See [pytorch.org](https://pytorch.org) for the correct command for your CUDA version. For example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## Quick Start

See `notebooks/examples/sfumato_full_pipeline.ipynb` for a complete walkthrough.

To run from the command line:

```bash
python run_sfumato.py --config configs/mousexeniumniche.json
```

Or run the two steps separately:

```bash
python preprocess.py --config configs/mousexeniumniche.json
python run_bgm_gpu.py --config configs/mousexeniumniche.json
```

## Main Config

Example config file: `configs/mousexeniumniche.json`

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

For example, with `k_list = [15, 20, 25, 30]` and `bgm_oversample = 1.5`:

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

For each K:

1. the shared dendrogram is cut at K;
2. each final cluster receives a colour according to the span of its oversampled BGM components along the optimal dendrogram leaf ordering;
3. finer K values inherit related colours from the same broader dendrogram branches.

Colourblind-friendly alternatives:

```json
"color": { "colormap": "viridis", "colormap_start": 0.0, "colormap_end": 1.0 }
```

## Outputs

### Preprocessing cache

```text
preprocess_cache_mousexeniumniche/
  mousexeniumniche_BIN40_F8_SVD10.npz
  mousexeniumniche_BIN40_F8_SVD10_preprocess_config.json
```

### Shared BGM outputs

```text
results_mousexeniumniche/
  shared_BGM/SVD10/
    *_cluster_color_table_allK.csv
    *_oversampled_merge_distances.csv
    *_shared_bgm_config.json
    weights_*.txt
    images/
      *_multiresolution_dendrogram_allK.png
```

### K-specific outputs

```text
results_mousexeniumniche/
  K30/SVD10/
    *_FULL.h5ad
    *_binlevel_FULL.csv
    colors_*.csv
    weights_*.txt
    *_bgm_config.json
    images/
      *_cut_dendrogram.png
```

### Column legend — `_binlevel_FULL.csv`

| Column | Description |
|--------|-------------|
| `x`, `y` | Bin spatial coordinates |
| `bin_id` | Global bin index |
| `cluster` | Index of the dominant cluster (0-based) |
| `second_cluster` | Index of the second-best cluster |
| `p1` | Probability of the dominant cluster |
| `p2` | Probability of the second-best cluster |
| `compl_p1` | `1 − p1` — uncertainty (0 = certain, ~1 = ambiguous) |
| `mix_t` | Blend weight toward second cluster: `log1p(α·p2) / (log1p(α·p1) + log1p(α·p2))` |
| `cmap_pos` | Cluster position along the dendrogram colour axis [0, 1] |
| `color_hard` | Hex colour of the dominant cluster |
| `color_mixed` | Hex colour blending top-2 clusters by log-probability — **recommended for visualisation** |

The transcript-level probability CSV is skipped by default. Enable with:

```json
"save_transcript_proba": true
```

## Dendrogram Figures

**Shared multiresolution dendrogram** (`*_multiresolution_dendrogram_allK.png`):

- shared oversampled-component dendrogram
- branches coloured across resolutions
- one dashed cut line per K
- aligned colour strips at each K

**K-specific dendrogram** (`*_cut_dendrogram.png`):

- same shared dendrogram
- branches coloured for the selected K
- one dashed cut line and one colour strip

## Optional UMAP Figures

```bash
python plots_umap/plot_embedding_umap.py --config configs/mousexeniumniche.json
python plots_umap/plot_embedding_umap.py --config configs/mousexeniumniche.json --k 30
```

## Optional Embedding-Dimension Figures

```bash
python plots_umap/plot_embedding_dims.py --config configs/mousexeniumniche.json --k 30
```

## Mask Generation

`make_masks/` contains scripts to generate semantic spatial masks from the bin-level posterior outputs. These were used for the paper analyses and require additional dependencies (`opencv-python`).

## HPC / SLURM

```bash
bash sbatch/submit_pipeline.sh
```

## Notes

- `X_norm` is the sparse-SVD embedding used by BGM.
- BGM is fitted once using `K_bgm = ceil(bgm_oversample * max(k_list))`.
- All requested K values are obtained by cutting the same shared dendrogram.
- Cluster colours are assigned from the shared dendrogram ordering, not from a separate PCA of final centroids.
- Colour differences should be interpreted as following the shared hierarchical ordering, not as exact metric distances in expression space.

## Authors

Alessia Giustolisi\*, Christophe Avenel\*†, Carolina Wählby\*†

\* Department of Information Technology, Uppsala University, Uppsala, Sweden  
† BioImage Informatics Facility, Science for Life Laboratory (SciLifeLab), Sweden

Contact me: alessia.giustolisi@it.uu.se

**Contact person for this repo:** alessia.giustolisi@it.uu.se

## Citation

Coming soon

## License
This project is licensed under the MIT License. See the LICENSE file for details.
