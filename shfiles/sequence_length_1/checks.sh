#!/bin/bash

echo "===================================="
echo "Job started on: $(date)"
echo "Running on node: $(hostname)"
echo "===================================="

# -----------------------------------
# Load module system
# ----------------------------------- 
source /etc/profile
source /etc/profile.d/modules.sh

module purge
module load conda
module load cuda/12.8

# -----------------------------------
# Activate conda environment
# -----------------------------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cancer_work

# -----------------------------------
# Force correct environment paths
# -----------------------------------
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Optional CUDA optimization
export CUDA_HOME=/usr/local/cuda-12.8

echo "===================================="
echo "Active Conda Env : $CONDA_DEFAULT_ENV"
echo "Python Path      : $(which python)"
echo "Python Version   : $(python --version)"
echo "===================================="

# -----------------------------------
# Project paths
# -----------------------------------
ROOT_DIR="/home/osharma2/cancer_workss/Repository_Breast_Tumor_Cells/data/"
PROJECT_DIR="/home/osharma2/cancer_workss/Repository_Breast_Tumor_Cells"
RESULTS_DIR="$PROJECT_DIR/results"

mkdir -p "$RESULTS_DIR"

# -----------------------------------
# Environment verification
# -----------------------------------
echo "Running environment verification..."


cd "$PROJECT_DIR"

echo "Starting training..."
echo "===================================="
# -----------------------------------
# Run training
# -----------------------------------
python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname alexnet_Frozen \
  --num_layers 3 \
  --channels 1 \
  --batch_size 8 \
  --epochs 1 \
  --lr 0.001 \
  --normalize 0 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 1 \
  --sequence_length 1 \
  --stride 1 \
  --seed 42 \
  --framesize 64 \
  --itr 1

python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname alexnet_Frozen \
  --num_layers 3 \
  --channels 1 \
  --batch_size 8 \
  --epochs 2 \
  --lr 0.001 \
  --normalize 0 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 2 \
  --sequence_length 1 \
  --stride 1 \
  --seed 42 \
  --framesize 64 \
  --itr 1
echo "===================================="
echo "Job finished on: $(date)"
echo "===================================="