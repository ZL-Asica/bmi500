#!/bin/bash -l
#SBATCH --job-name=bmi500-pbmc3k
#SBATCH --partition=overflow
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_outputs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source /usr/share/Modules/init/bash
module load bmi/python-3.12.12
source .venv-cluster312/bin/activate
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMBA_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export NUMBA_CACHE_DIR="$PWD/.cache/numba"
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" cluster-results/pbmc3k
echo "JOB=$SLURM_JOB_ID HOST=$(hostname) DATASET=pbmc3k"
date -Is
/usr/bin/time -v -o cluster-results/pbmc3k/time-v.txt \
  python -u scanpy_pbmc.py --data-dir data --data-set pbmc3k \
  --out-dir cluster-results/pbmc3k --num-threads "$SLURM_CPUS_PER_TASK" \
  --profile-output cluster-results/pbmc3k/rank_genes_groups.prof \
  > cluster-results/pbmc3k/console.txt 2>&1
cat cluster-results/pbmc3k/console.txt
cat cluster-results/pbmc3k/time-v.txt
