#!/usr/bin/env bash
#SBATCH -A NAISS2026-4-793
#SBATCH --gpus-per-node=V100:1
#SBATCH --cpus-per-task=8
#SBATCH -t 0-01:00:00
#SBATCH -o bgm_gpu_svd.%j.out
#SBATCH -e bgm_gpu_svd.%j.err

set -euo pipefail

WORKDIR=/mimer/NOBACKUP/groups/digital_pathology/bayesian_clustering
IMAGE=${WORKDIR}/bgm-gpu_v1.sif
CONFIG=/workspace/configs/mousexeniumniche_diffseed.json

apptainer exec --nv \
    --bind ${WORKDIR}:/workspace \
    ${IMAGE} \
    bash -lc "cd /workspace && python -u /workspace/bgm_gpu_step_svd.py --config ${CONFIG}"
