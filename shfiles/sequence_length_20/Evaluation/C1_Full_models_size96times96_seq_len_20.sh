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

PROJECT_DIR="/home/osharma2/cancer_workss/Repository_Breast_Tumor_Cells"
ROOT_DIR="$PROJECT_DIR/data/"
RESULTS_DIR="$PROJECT_DIR/results"

mkdir -p "$RESULTS_DIR"

# -----------------------------------
# Environment verification
# -----------------------------------
echo "Running environment verification..."


cd "$PROJECT_DIR"

echo "Starting training..."
echo "===================================="



python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C1_cnn2d \
  --num_layers 3 \
  --channels 16 32 64 \
  --batch_size 16 \
  --epochs 1 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 2 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1

python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C1_alexnet \
  --num_layers 3 \
  --channels 1 \
  --batch_size 16 \
  --epochs 0 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 2 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 

python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C1_vgg16 \
  --num_layers 3 \
  --channels 1 \
  --batch_size 16 \
  --epochs 0 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 2 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 

python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C1_vgg19 \
  --num_layers 3 \
  --channels 1 \
  --batch_size 16 \
  --epochs 0 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 2 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 


python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C1_resnet18 \
  --num_layers 3 \
  --channels 1 \
  --batch_size 16 \
  --epochs 0 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 2 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 

python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C1_resnet34 \
  --num_layers 3 \
  --channels 1 \
  --batch_size 16 \
  --epochs 0 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 2 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 

python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C1_resnet50 \
  --num_layers 3 \
  --channels 1 \
  --batch_size 16 \
  --epochs 0 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 2 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 

python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C1_resnet101 \
  --num_layers 3 \
  --channels 1 \
  --batch_size 16 \
  --epochs 0 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 2 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 


python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C1_densenet121 \
  --num_layers 3 \
  --channels 1 \
  --batch_size 16 \
  --epochs 0 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 2 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 


python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C1_densenet169 \
  --num_layers 3 \
  --channels 1 \
  --batch_size 16 \
  --epochs 0 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 2 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 


python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C1_densenet201 \
  --num_layers 3 \
  --channels 1 \
  --batch_size 16 \
  --epochs 0 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 2 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 



python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C1_efficientnet_b0 \
  --num_layers 3 \
  --channels 1 \
  --batch_size 16 \
  --epochs 0 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 2 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 


 
python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C1_efficientnet_b1 \
  --num_layers 3 \
  --channels 1 \
  --batch_size 16 \
  --epochs 0 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 2 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 


python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C1_efficientnet_b2 \
  --num_layers 3 \
  --channels 1 \
  --batch_size 16 \
  --epochs 0 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 2 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 


 
python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C1_efficientnet_b3 \
  --num_layers 3 \
  --channels 1 \
  --batch_size 16 \
  --epochs 0 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 2 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 


python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C1_efficientnet_b4 \
  --num_layers 3 \
  --channels 1 \
  --batch_size 16 \
  --epochs 0 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 2 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 


 
python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C1_efficientnet_b5 \
  --num_layers 3 \
  --channels 1 \
  --batch_size 16 \
  --epochs 0 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 2 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 


python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C1_efficientnet_b6 \
  --num_layers 3 \
  --channels 1 \
  --batch_size 16 \
  --epochs 0 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 2 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 


 
python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C1_efficientnet_b7 \
  --num_layers 3 \
  --channels 1 \
  --batch_size 16 \
  --epochs 0 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 2 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 

python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C1_unet \
  --num_layers 3 \
  --channels 1 \
  --batch_size 16 \
  --epochs 0 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 2 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 

python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C1_unetpp \
  --num_layers 3 \
  --channels 1 \
  --batch_size 16 \
  --epochs 0 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 2 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 



python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C1_mobilenet_v2 \
  --num_layers 3 \
  --channels 1 \
  --batch_size 16 \
  --epochs 0 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 2 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 


 
python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C1_mobilenet_v3_small \
  --num_layers 3 \
  --channels 1 \
  --batch_size 16 \
  --epochs 0 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 2 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 


python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C1_mobilenet_v3_large \
  --num_layers 3 \
  --channels 1 \
  --batch_size 16 \
  --epochs 0 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 2 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 


 
python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C1_vit \
  --num_layers 3 \
  --channels 1 \
  --batch_size 16 \
  --epochs 0 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 2 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 


python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C1_swin \
  --num_layers 3 \
  --channels 1 \
  --batch_size 16 \
  --epochs 0 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 2 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 


 
python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C1_attention_unet \
  --num_layers 3 \
  --channels 1 \
  --batch_size 16 \
  --epochs 0 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 2 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 

python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C1_deeplabv3 \
  --num_layers 3 \
  --channels 1 \
  --batch_size 16 \
  --epochs 0 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 2 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 

python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C1_hrnet \
  --num_layers 3 \
  --channels 1 \
  --batch_size 16 \
  --epochs 0 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 2 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 

python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C1_segformer \
  --num_layers 3 \
  --channels 1 \
  --batch_size 16 \
  --epochs 0 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 2 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 

python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C1_unet_resnet \
  --num_layers 3 \
  --channels 1 \
  --batch_size 16 \
  --epochs 0 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 2 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 

python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C1_attention_unet_eff \
  --num_layers 3 \
  --channels 1 \
  --batch_size 16 \
  --epochs 0 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 2 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 


python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C1_cellpose \
  --num_layers 3 \
  --channels 1 \
  --batch_size 16 \
  --epochs 0 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 2 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 





