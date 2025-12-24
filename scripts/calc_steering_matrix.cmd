@echo off
setlocal

REM =========================
REM Configuration
REM =========================

set TRAIN_VAL_DIR=data\instructions\train_val
set EMBEDDING_DIR=data\embeddings\llama3.1
set NICKNAME=llama3.1
set MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
set DEVICE=cuda:0

REM =========================
REM Calculate steering matrix
REM =========================

set STEERING_SAVE_PATH=data\steering_matrix\steering_matrix_%NICKNAME%.pt
echo Calculating steering matrix for %NICKNAME%

python src\calc_steering_matrix.py ^
    --model_name "%NICKNAME%" ^
    --embedding_dir "%EMBEDDING_DIR%" ^
    --device "%DEVICE%" ^
    --save_path "%STEERING_SAVE_PATH%"

echo Done.
endlocal
