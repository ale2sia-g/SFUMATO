#!/usr/bin/env bash
#SBATCH -A NAISS2026-4-793
#SBATCH --gpus-per-node=A40:1
#SBATCH --cpus-per-task=4
#SBATCH -t 0-01:00:00
#SBATCH -o bgm_gpu.%j.out
#SBATCH -e bgm_gpu.%j.err

set -euo pipefail

WORKDIR=/mimer/NOBACKUP/groups/digital_pathology/bayesian_clustering
IMAGE=${WORKDIR}/bgm-gpu_v1.sif
CONFIG=/workspace/configs/mousexeniumcell.json

apptainer exec --nv \
    --bind ${WORKDIR}:/workspace \
    ${IMAGE} \
    bash -lc "cd /workspace && export PYTHONPATH=/workspace && python -u /workspace/run_bgm_gpu.py --config ${CONFIG}"