#!/usr/bin/env bash
set -euo pipefail

JOB1=$(sbatch --parsable sbatch/preprocess.sbatch)
echo "SVD preprocessing job: ${JOB1}"

JOB2=$(sbatch --parsable --dependency=afterok:${JOB1} sbatch/run_bgm_gpu.sbatch)
echo "SVD BGM GPU job: ${JOB2}"

echo "SVD pipeline submitted. Job 2 will start automatically after Job 1 completes."
