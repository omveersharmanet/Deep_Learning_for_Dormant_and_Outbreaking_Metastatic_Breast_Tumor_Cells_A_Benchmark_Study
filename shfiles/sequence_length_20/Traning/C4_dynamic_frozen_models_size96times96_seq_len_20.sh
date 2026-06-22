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
  --modelname C4_alexnet_Dynamic_Frozen \
  --num_layers 10 \
  --num_lastlayerstrain 6 \
  --channels 1 \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 1 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 

python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C4_vgg16_Dynamic_Frozen \
  --num_layers 14 \
  --num_lastlayerstrain 10 \
  --channels 1 \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 1 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 

python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C4_vgg19_Dynamic_Frozen \
  --num_layers 14 \
  --num_lastlayerstrain 10 \
  --channels 1 \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 1 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 

python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C4_resnet18_Dynamic_Frozen \
  --num_layers 3 \
  --num_lastlayerstrain 2 \
  --channels 1 \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 1 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 

python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C4_resnet34_Dynamic_Frozen \
  --num_layers 3 \
  --num_lastlayerstrain 2 \
  --channels 1 \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 1 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 

python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C4_resnet50_Dynamic_Frozen \
  --num_layers 3 \
  --num_lastlayerstrain 2 \
  --channels 1 \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 1 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 

python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C4_resnet101_Dynamic_Frozen \
  --num_layers 3 \
  --num_lastlayerstrain 2 \
  --channels 1 \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 1 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 

python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C4_densenet121_Dynamic_Frozen \
  --num_layers 2 \
  --num_lastlayerstrain 1 \
  --channels 1 \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 1 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 

python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C4_densenet169_Dynamic_Frozen \
  --num_layers 2 \
  --num_lastlayerstrain 1 \
  --channels 1 \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 1 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 

python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C4_densenet201_Dynamic_Frozen \
  --num_layers 2 \
  --num_lastlayerstrain 1 \
  --channels 1 \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 1 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 



python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C4_efficientnet_b0_Dynamic_Frozen \
  --num_layers 2 \
  --num_lastlayerstrain 1 \
  --channels 1 \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 1 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 


python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C4_efficientnet_b1_Dynamic_Frozen \
  --num_layers 2 \
  --num_lastlayerstrain 1 \
  --channels 1 \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 1 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 


python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C4_efficientnet_b2_Dynamic_Frozen \
  --num_layers 2 \
  --num_lastlayerstrain 1 \
  --channels 1 \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 1 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 


python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C4_efficientnet_b3_Dynamic_Frozen \
  --num_layers 2 \
  --num_lastlayerstrain 1 \
  --channels 1 \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 1 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 


python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C4_efficientnet_b4_Dynamic_Frozen \
  --num_layers 2 \
  --num_lastlayerstrain 1 \
  --channels 1 \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 1 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 


python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C4_efficientnet_b5_Dynamic_Frozen \
  --num_layers 2 \
  --num_lastlayerstrain 1 \
  --channels 1 \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 1 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 


python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C4_efficientnet_b6_Dynamic_Frozen \
  --num_layers 2 \
  --num_lastlayerstrain 1 \
  --channels 1 \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 1 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 


python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C4_efficientnet_b7_Dynamic_Frozen \
  --num_layers 2 \
  --num_lastlayerstrain 1 \
  --channels 1 \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 1 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 


python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C4_unet_Dynamic_Frozen \
  --num_layers 2 \
  --num_lastlayerstrain 1 \
  --channels 1 \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 1 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 

python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C4_unetpp_Dynamic_Frozen \
  --num_layers 2 \
  --num_lastlayerstrain 1 \
  --channels 1 \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 1 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 




python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C4_mobilenet_v2_Dynamic_Frozen \
  --num_layers 3 \
  --num_lastlayerstrain 2 \
  --channels 1 \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 1 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 

python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C4_mobilenet_v3_small_Dynamic_Frozen \
  --num_layers 3 \
  --num_lastlayerstrain 2 \
  --channels 1 \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 1 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 


python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C4_mobilenet_v3_large_Dynamic_Frozen \
  --num_layers 3 \
  --num_lastlayerstrain 2 \
  --channels 1 \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 1 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 

python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C4_vit_Dynamic_Frozen \
  --num_layers 2 \
  --num_lastlayerstrain 1 \
  --channels 1 \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 1 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 


python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C4_swin_Dynamic_Frozen \
  --num_layers 2 \
  --num_lastlayerstrain 1 \
  --channels 1 \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 1 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 

python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C4_attention_unet_Dynamic_Frozen \
  --num_layers 1 \
  --num_lastlayerstrain 0 \
  --channels 1 \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 1 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 


python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C4_deeplabv3_Dynamic_Frozen \
  --num_layers 1 \
  --num_lastlayerstrain 0 \
  --channels 1 \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 1 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 

python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C4_hrnet_Dynamic_Frozen \
  --num_layers 1 \
  --num_lastlayerstrain 0 \
  --channels 1 \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 1 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 

python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C4_segformer_Dynamic_Frozen \
  --num_layers 2 \
  --num_lastlayerstrain 1 \
  --channels 1 \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 1 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 

python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C4_unet_resnet_Dynamic_Frozen \
  --num_layers 2 \
  --num_lastlayerstrain 1 \
  --channels 1 \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 1 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 

python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C4_attention_unet_eff_Dynamic_Frozen \
  --num_layers 2 \
  --num_lastlayerstrain 1 \
  --channels 1 \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 1 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 

python main.py \
  --root "$ROOT_DIR" \
  --results_dir "$RESULTS_DIR" \
  --modelname C4_cellpose_Dynamic_Frozen \
  --num_layers 2 \
  --num_lastlayerstrain 1 \
  --channels 1 \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.0005 \
  --dropout 0.1 \
  --patience_iter 10 \
  --early_stop_iter 13 \
  --is_traning 1 \
  --sequence_length 20 \
  --framesize 96 \
  --itr 1 