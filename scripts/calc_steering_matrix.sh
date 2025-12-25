#!/bin/bash

TRAIN_VAL_DIR=data/instructions/train_val
# Configuration - Change these variables to test different models and output directories
EMBEDDING_DIR=data/embeddings/qwen2.5  # Output directory for embeddings
NICKNAME=qwen2.5
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct  # Model name from HuggingFace

DEVICE=cuda:0

# Calculate steering matrix
STEERING_SAVE_PATH=data/steering_matrix/steering_matrix_${NICKNAME}.pt
echo "Calculating steering matrix for $NICKNAME"
python src/calc_steering_matrix.py --model_name $NICKNAME\
                                    --embedding_dir $EMBEDDING_DIR \
                                    --device $DEVICE \
                                    --save_path $STEERING_SAVE_PATH