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
  --modelname C2_alexnet_Frozen \
  --num_layers 3 \
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
  --modelname C2_vgg16_Frozen \
  --num_layers 3 \
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
  --modelname C2_vgg19_Frozen \
  --num_layers 3 \
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
  --modelname C2_resnet18_Frozen \
  --num_layers 3 \
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
  --modelname C2_resnet34_Frozen \
  --num_layers 3 \
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
  --modelname C2_resnet50_Frozen \
  --num_layers 3 \
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
  --modelname C2_resnet101_Frozen \
  --num_layers 3 \
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
  --modelname C2_densenet121_Frozen \
  --num_layers 3 \
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
  --modelname C2_densenet169_Frozen \
  --num_layers 3 \
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
  --modelname C2_densenet201_Frozen \
  --num_layers 3 \
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
  --modelname C2_efficientnet_b0_Frozen \
  --num_layers 3 \
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
  --modelname C2_efficientnet_b1_Frozen \
  --num_layers 3 \
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
  --modelname C2_efficientnet_b2_Frozen \
  --num_layers 3 \
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
  --modelname C2_efficientnet_b3_Frozen \
  --num_layers 3 \
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
  --modelname C2_efficientnet_b4_Frozen \
  --num_layers 3 \
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
  --modelname C2_efficientnet_b5_Frozen \
  --num_layers 3 \
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
  --modelname C2_efficientnet_b6_Frozen \
  --num_layers 3 \
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
  --modelname C2_efficientnet_b7_Frozen \
  --num_layers 3 \
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
  --modelname C2_unet_Frozen \
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
  --modelname C2_unetpp_Frozen \
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
  --modelname C2_mobilenet_v2_Frozen \
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
  --modelname C2_mobilenet_v3_small_Frozen \
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
  --modelname C2_mobilenet_v3_large_Frozen \
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
  --modelname C2_vit_Frozen \
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
  --modelname C2_swin_Frozen \
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
  --modelname C2_attention_unet_Frozen \
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
  --modelname C2_deeplabv3_Frozen \
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
  --modelname C2_hrnet_Frozen \
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
  --modelname C2_segformer_Frozen \
  --num_layers 1 \
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
  --modelname C2_unet_resnet_Frozen \
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
  --modelname C2_attention_unet_eff_Frozen \
  --num_layers 1 \
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
  --modelname C2_cellpose_Frozen \
  --num_layers 1 \
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

