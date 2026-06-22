# Deep_Learning_for_Dormant_and_Outbreaking_Metastatic_Breast_Tumor_Cells_A_Benchmark_Study
Deep learning benchmark for classifying dormant and proliferative metastatic breast tumor cells in a 3D BME model. Evaluates CNNs, segmentation networks, and transformers with transfer learning and temporal sequence modeling. EfficientNet-B7 achieved the best performance with 98.86% accuracy and ROC-AUC of 0.998.

## 🌟 Getting Start

### 🛠️ Installation

```bash
pip install -r requirements.txt
```

### 📦 Datasets



### 🚀 Train and evaluate
#### Evaluation Top models only 
```bash
bash ./shfiles/C3_.sh 
```

#### Train
```bash
# Imagesize 64x64 configuration C1 to C4 sequence length 1 --> Training 
bash ./shfiles/sequence_length_1/Training/C1_Full_models_size64times64_seq_len_1.sh 
bash ./shfiles/sequence_length_1/Training/C2_full_frozen_models_size64times64_seq_len_1.sh 
bash ./shfiles/sequence_length_1/Training/C3_dynamic_frozen_with_dynamic_train_size64times64_seq_len_1.sh 
bash ./shfiles/sequence_length_1/Training/C4_dynamic_frozen_models_size64times64_seq_len_1.sh

# Imagesize 96x96 configuration C1 to C4 sequence length 1 --> Training

bash ./shfiles/sequence_length_1/Training/C1_Full_models_size96times96_seq_len_1.sh 
bash ./shfiles/sequence_length_1/Training/C2_full_frozen_models_size96times96_seq_len_1.sh 
bash ./shfiles/sequence_length_1/Training/C3_dynamic_frozen_with_dynamic_train_size96times96_seq_len_1.sh 
bash ./shfiles/sequence_length_1/Training/C4_dynamic_frozen_models_size96times96_seq_len_1.sh 

# Imagesize 64x64 configuration C1 to C4 sequence length 3 --> Training 
bash ./shfiles/sequence_length_3/Training/C1_Full_models_size64times64_seq_len_3.sh 
bash ./shfiles/sequence_length_3/Training/C2_full_frozen_models_size64times64_seq_len_3.sh 
bash ./shfiles/sequence_length_3/Training/C3_dynamic_frozen_with_dynamic_train_size64times64_seq_len_3.sh 
bash ./shfiles/sequence_length_3/Training/C4_dynamic_frozen_models_size64times64_seq_len_3.sh

# Imagesize 96x96 configuration C1 to C4 sequence length 3 --> Training
bash ./shfiles/sequence_length_3/Training/C1_Full_models_size96times96_seq_len_3.sh 
bash ./shfiles/sequence_length_3/Training/C2_full_frozen_models_size96times96_seq_len_3.sh 
bash ./shfiles/sequence_length_3/Training/C3_dynamic_frozen_with_dynamic_train_size96times96_seq_len_3.sh 
bash ./shfiles/sequence_length_3/Training/C4_dynamic_frozen_models_size96times96_seq_len_3.sh 

# Imagesize 64x64 configuration C1 to C4 sequence length 10 --> Training 
bash ./shfiles/sequence_length_10/Training/C1_Full_models_size64times64_seq_len_10.sh 
bash ./shfiles/sequence_length_10/Training/C2_full_frozen_models_size64times64_seq_len_10.sh 
bash ./shfiles/sequence_length_10/Training/C3_dynamic_frozen_with_dynamic_train_size64times64_seq_len_10.sh 
bash ./shfiles/sequence_length_10/Training/C4_dynamic_frozen_models_size64times64_seq_len_10.sh

# Imagesize 96x96 configuration C1 to C4 sequence length 10 --> Training
bash ./shfiles/sequence_length_10/Training/C1_Full_models_size96times96_seq_len_10.sh 
bash ./shfiles/sequence_length_10/Training/C2_full_frozen_models_size96times96_seq_len_10.sh 
bash ./shfiles/sequence_length_10/Training/C3_dynamic_frozen_with_dynamic_train_size96times96_seq_len_10.sh 
bash ./shfiles/sequence_length_10/Training/C4_dynamic_frozen_models_size96times96_seq_len_10.sh 

# Imagesize 64x64 configuration C1 to C4 sequence length 20 --> Training 
bash ./shfiles/sequence_length_20/Training/C1_Full_models_size64times64_seq_len_20.sh 
bash ./shfiles/sequence_length_20/Training/C2_full_frozen_models_size64times64_seq_len_20.sh 
bash ./shfiles/sequence_length_20/Training/C3_dynamic_frozen_with_dynamic_train_size64times64_seq_len_20.sh 
bash ./shfiles/sequence_length_20/Training/C4_dynamic_frozen_models_size64times64_seq_len_20.sh

# Imagesize 96x96 configuration C1 to C4 sequence length 20 --> Training
bash ./shfiles/sequence_length_20/Training/C1_Full_models_size96times96_seq_len_20.sh 
bash ./shfiles/sequence_length_20/Training/C2_full_frozen_models_size96times96_seq_len_20.sh 
bash ./shfiles/sequence_length_20/Training/C3_dynamic_frozen_with_dynamic_train_size96times96_seq_len_20.sh 
bash ./shfiles/sequence_length_20/Training/C4_dynamic_frozen_models_size96times96_seq_len_20.sh 
```
#### Evaluation
```bash
# Imagesize 64x64 configuration C1 to C4 sequence length 1 --> Evaluation
bash ./shfiles/sequence_length_1/Evaluation/C1_Full_models_size64times64_seq_len_1.sh 
bash ./shfiles/sequence_length_1/Evaluation/C2_full_frozen_models_size64times64_seq_len_1.sh 
bash ./shfiles/sequence_length_1/Evaluation/C3_dynamic_frozen_with_dynamic_train_size64times64_seq_len_1.sh 
bash ./shfiles/sequence_length_1/Evaluation/C4_dynamic_frozen_models_size64times64_seq_len_1.sh

# Imagesize 96x96 configuration C1 to C4 sequence length 1 --> Training

bash ./shfiles/sequence_length_1/Evaluation/C1_Full_models_size96times96_seq_len_1.sh 
bash ./shfiles/sequence_length_1/Evaluation/C2_full_frozen_models_size96times96_seq_len_1.sh 
bash ./shfiles/sequence_length_1/Evaluation/C3_dynamic_frozen_with_dynamic_train_size96times96_seq_len_1.sh 
bash ./shfiles/sequence_length_1/Evaluation/C4_dynamic_frozen_models_size96times96_seq_len_1.sh 

# Imagesize 64x64 configuration C1 to C4 sequence length 3 --> Evaluation
bash ./shfiles/sequence_length_3/Evaluation/C1_Full_models_size64times64_seq_len_3.sh 
bash ./shfiles/sequence_length_3/Evaluation/C2_full_frozen_models_size64times64_seq_len_3.sh 
bash ./shfiles/sequence_length_3/Evaluation/C3_dynamic_frozen_with_dynamic_train_size64times64_seq_len_3.sh 
bash ./shfiles/sequence_length_3/Evaluation/C4_dynamic_frozen_models_size64times64_seq_len_3.sh

# Imagesize 96x96 configuration C1 to C4 sequence length 3 --> Training
bash ./shfiles/sequence_length_3/Evaluation/C1_Full_models_size96times96_seq_len_3.sh 
bash ./shfiles/sequence_length_3/Evaluation/C2_full_frozen_models_size96times96_seq_len_3.sh 
bash ./shfiles/sequence_length_3/Evaluation/C3_dynamic_frozen_with_dynamic_train_size96times96_seq_len_3.sh 
bash ./shfiles/sequence_length_3/Evaluation/C4_dynamic_frozen_models_size96times96_seq_len_3.sh 

# Imagesize 64x64 configuration C1 to C4 sequence length 10 --> Evaluation
bash ./shfiles/sequence_length_10/Evaluation/C1_Full_models_size64times64_seq_len_10.sh 
bash ./shfiles/sequence_length_10/Evaluation/C2_full_frozen_models_size64times64_seq_len_10.sh 
bash ./shfiles/sequence_length_10/Evaluation/C3_dynamic_frozen_with_dynamic_train_size64times64_seq_len_10.sh 
bash ./shfiles/sequence_length_10/Evaluation/C4_dynamic_frozen_models_size64times64_seq_len_10.sh

# Imagesize 96x96 configuration C1 to C4 sequence length 10 --> Training
bash ./shfiles/sequence_length_10/Evaluation/C1_Full_models_size96times96_seq_len_10.sh 
bash ./shfiles/sequence_length_10/Evaluation/C2_full_frozen_models_size96times96_seq_len_10.sh 
bash ./shfiles/sequence_length_10/Evaluation/C3_dynamic_frozen_with_dynamic_train_size96times96_seq_len_10.sh 
bash ./shfiles/sequence_length_10/Evaluation/C4_dynamic_frozen_models_size96times96_seq_len_10.sh 

# Imagesize 64x64 configuration C1 to C4 sequence length 20 --> Evaluation
bash ./shfiles/sequence_length_20/Evaluation/C1_Full_models_size64times64_seq_len_20.sh 
bash ./shfiles/sequence_length_20/Evaluation/C2_full_frozen_models_size64times64_seq_len_20.sh 
bash ./shfiles/sequence_length_20/Evaluation/C3_dynamic_frozen_with_dynamic_train_size64times64_seq_len_20.sh 
bash ./shfiles/sequence_length_20/Evaluation/C4_dynamic_frozen_models_size64times64_seq_len_20.sh

# Imagesize 96x96 configuration C1 to C4 sequence length 20 --> Training
bash ./shfiles/sequence_length_20/Evaluation/C1_Full_models_size96times96_seq_len_20.sh 
bash ./shfiles/sequence_length_20/Evaluation/C2_full_frozen_models_size96times96_seq_len_20.sh 
bash ./shfiles/sequence_length_20/Evaluation/C3_dynamic_frozen_with_dynamic_train_size96times96_seq_len_20.sh 
bash ./shfiles/sequence_length_20/Evaluation/C4_dynamic_frozen_models_size96times96_seq_len_20.sh 

```


## :pray: Acknowledgement 

This work was supported by the Artificial Intelligence Research Center at the University of Haifa.


## 🤝 Join the Collaboration
We warmly welcome your participation! Whether you have ideas for improvements, feature additions, or bug fixes, feel free to open an issue or submit a pull request.


