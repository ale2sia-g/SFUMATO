#!/bin/bash
#SBATCH -A NAISS2026-4-793
#SBATCH -C NOGPU
#SBATCH -t 04:00:00
#SBATCH -J cluster_masks
#SBATCH -o logs/cluster_masks_bin7_%j.out
#SBATCH -e logs/cluster_masks_bin7_%j.err
#SBATCH --cpus-per-task=32

set -euo pipefail

WORKDIR=/mimer/NOBACKUP/groups/digital_pathology/bayesian_clustering
IMAGE=${WORKDIR}/bgm-gpu_v1.sif

mkdir -p ${WORKDIR}/logs

SCRIPT=/workspace/make_cluster_masks_from_posterior.py
CSV=/workspace/results_mousexeniumcell/K45/SVD10/mousexeniumcell_BGM68to45_BIN7_F5_SVD10_transcripts_proba_FULL.csv
OUTDIR=/workspace/results_mousexeniumcell/K45/SVD10/cluster_masks_K45_BIN7_cell_level_SVD10

apptainer exec \
    --bind ${WORKDIR}:/workspace \
    ${IMAGE} \
    python make_cluster_masks_from_binlevel_posterior.py \
    --bin-csv results_mousexeniumcell/K45/SVD10/mousexeniumcell_BGM68to45_BIN7_F5_SVD10_binlevel_FULL.csv \
    --transcript-proba-csv results_mousexeniumcell/K45/SVD10/mousexeniumcell_BGM68to45_BIN7_F5_SVD10_transcripts_proba_FULL.csv \
    --image-size 48352 33135 \
    --pixel-size-um 0.2125 \
    --out-dir cluster_masks_K45_BIN7_cell_level_SVD10_binlevel_low010_high030 \
    --bin-size 7 \
    --sigma 7 \
    --low-threshold 0.10 \
    --high-threshold 0.30 \
    --threshold-scale relative