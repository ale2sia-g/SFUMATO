#!/usr/bin/env bash
#SBATCH -A NAISS2026-4-793
#SBATCH -C NOGPU
#SBATCH --cpus-per-task=32
#SBATCH -t 0-01:30:00
#SBATCH -o preprocess_svd.%j.out
#SBATCH -e preprocess_svd.%j.err

set -euo pipefail

WORKDIR=/mimer/NOBACKUP/groups/digital_pathology/bayesian_clustering
IMAGE=${WORKDIR}/bgm-gpu_v1.sif
CONFIG=/workspace/configs/mousexeniumniche_diffseed.json

apptainer exec \
    --bind ${WORKDIR}:/workspace \
    ${IMAGE} \
    bash -lc "cd /workspace && python -u /workspace/preprocess.py --config ${CONFIG}"
