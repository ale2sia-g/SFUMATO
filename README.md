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

This pipeline separates dataset/run configuration from reusable method code.  
The same config file is used for local runs, separate preprocessing/BGM runs, optional post-processing plots, and Alvis SLURM jobs.

## Main Config

Example for niche-level Xenium Mouse Brain:

```bash
configs/mousexeniumniche.json
```

Example run settings:

- `run_name`: `mousexeniumniche`
- `input_csv`: `xenium_mouse.csv`
- `k_list`: `[15, 20, 25, 30]`
- `bgm_oversample`: `1.5`
- `min_genes_per_bin`: `120`
- `bin_width`: `40`
- `factor`: `8`
- `seed`: `8`
- `use_svd`: `true`
- `use_pca`: `false`
- `comp`: `10`
- `save_p2r`: `true`
- `save_transcript_proba`: `false`

## Dimensionality Reduction

SFUMATO supports sparse TruncatedSVD and dense PCA.

Use SVD only:

```json
"use_svd": true,
"use_pca": false
```

Use PCA only:

```json
"use_svd": false,
"use_pca": true
```

Run both SVD and PCA from the same preprocessing step:

```json
"use_svd": true,
"use_pca": true
```

When both are enabled, `points2regions_withsplit` is run once, then separate caches are written for each embedding method.

Use a base cache/output name in the config:

```json
"cache_dir": "preprocess_cache_mousexeniumniche"
```

```json
"outroot": "results_mousexeniumniche"
```

The code will create method-specific folders automatically:

```text
preprocess_cache_mousexeniumniche_svd/
preprocess_cache_mousexeniumniche_pca/
results_mousexeniumniche_svd/
results_mousexeniumniche_pca/
```

PCA requires densifying the binned feature matrix and can require much more RAM than SVD. Use PCA only for datasets small enough to fit in memory.

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

Overwrite existing preprocessing caches:

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

## Outputs

For SVD:

```text
preprocess_cache_mousexeniumniche_svd/
  mousexeniumniche_BIN40_F8_SVD10.npz
```

For PCA:

```text
preprocess_cache_mousexeniumniche_pca/
  mousexeniumniche_BIN40_F8_PCA10.npz
```

BGM outputs are grouped by method, `K`, and `COMP`.

SVD example:

```text
results_mousexeniumniche_svd/
  K15/
    SVD10/
      *_FULL.h5ad
      *_binlevel_FULL.csv
      hues_*.csv
      weights_*.txt
      *_bgm_config.json
      images/
        *_oversampled_dendrogram_cut.png
        *_final_dendrogram_colors.png
```

PCA example:

```text
results_mousexeniumniche_pca/
  K15/
    PCA10/
      *_FULL.h5ad
      *_binlevel_FULL.csv
      hues_*.csv
      weights_*.txt
      *_bgm_config.json
      images/
        *_oversampled_dendrogram_cut.png
        *_final_dendrogram_colors.png
```

The transcript-level probability CSV is skipped by default. Enable it with:

```json
"save_transcript_proba": true
```

## Optional Figures

UMAP post-processing:

```bash
python plot_embedding_umap.py --config configs/mousexeniumniche.json
```

For one K only:

```bash
python plot_embedding_umap.py --config configs/mousexeniumniche.json --k 30
```

If both SVD and PCA are enabled in the config, UMAP plots are generated for both methods.

Plot the first two embedding dimensions directly, without UMAP:

```bash
python plot_embedding_dims.py --config configs/mousexeniumniche.json
```

For one K only:

```bash
python plot_embedding_dims.py --config configs/mousexeniumniche.json --k 30
```

For PCA caches these are `PC1/PC2`.  
For SVD caches these are `SVD1/SVD2`.

Hue equalization figure from a `hues_*.csv` file:

```bash
python plot_hue_equalization.py \
  --hues-csv results_mousexeniumniche_svd/K30/SVD10/hues_mousexeniumniche_BGM45to30_BIN40_F8_SVD10.csv \
  --output results_mousexeniumniche_svd/K30/SVD10/images/mousexeniumniche_K30_SVD10_hue_equalization.png
```

Notebook versions are available in:

```text
notebooks/
```