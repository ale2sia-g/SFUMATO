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
The same config file is used for local runs, separate preprocessing/BGM runs, and Alvis SLURM jobs.

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
- `comp`: `10`
- `save_p2r`: `true`
- `save_transcript_proba`: `false`

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

Preprocessing cache:

```text
preprocess_cache_mousexeniumniche_svd/
  mousexeniumniche_BIN40_F8_SVD10.npz
```

BGM outputs are grouped by `K` and `COMP`:

```text
results_mousexeniumniche_svd/
  K15/
    SVD10/
      *_FULL.h5ad
      *_binlevel_FULL.csv
      hues_*.csv
      weights_*.txt
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
