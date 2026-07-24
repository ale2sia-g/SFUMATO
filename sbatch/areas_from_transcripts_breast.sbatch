#!/bin/bash
#SBATCH -A NAISS2026-4-793
#SBATCH -C NOGPU
#SBATCH -t 01:00:00
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
    python make_sfumato_semantic_masks_breast.py \
        --bin-csv results_breastxenium_niche/K13/SVD10/breastxenium_niche_BGM33to13_BIN40_F8_SVD10_binlevel_FULL.csv \
        --transcript-proba-csv results_breastxenium_niche/K13/SVD10/breastxenium_niche_BGM33to13_BIN40_F8_SVD10_transcripts_proba_FULL.csv \
        --annotation-geojson preprocess_xenium_data/Xenium_FFPE_Human_Breast_Cancer_Rep1_outs/outs/breast_annotations_fixed.geojson \
        --out-dir semantic_masks_breast_BIN40_F8 \
        --image-size 35416 25778 \
        --pixel-size-um 0.2125 \
        --invasive-clusters p5 p6 \
        --dcis-clusters p1 p2 p3 p4 \
        --bin-size 40 \
        --sigma 80 \
        --low-threshold 0.10 \
        --high-threshold 0.30