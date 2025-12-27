@echo off
setlocal EnableDelayedExpansion

REM =========================
REM Configuration
REM =========================

set TRAIN_VAL_DIR=data\instructions\train_val
set EMBEDDING_DIR=data\embeddings\TinyLlama
set NICKNAME=TinyLlama
set MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0
set DEVICE=cpu

REM =========================
REM Extract embeddings
REM =========================

for %%F in ("%TRAIN_VAL_DIR%\*.json") do (
    set "FILE=%%F"
    set "FILENAME=%%~nF"

    echo Extracting embeddings for %%F

    REM Decide prompt_column based on filename
    echo !FILENAME! | findstr /i "coconot" >nul
    if errorlevel 1 (
        set PROMPT_COLUMN=query
    ) else (
        set PROMPT_COLUMN=prompt
    )

    python src\extract_embeddings.py ^
        --model_name "%MODEL_NAME%" ^
        --input_file "%%F" ^
        --prompt_column "!PROMPT_COLUMN!" ^
        --output_file "%EMBEDDING_DIR%\embeds_!FILENAME!.pt" ^
        --batch_size 16 ^
        --device "%DEVICE%"
)

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

REM =========================
REM Steering / Generation
REM =========================

set GENERATE_CONFIG_DIR=config\llama3.1
echo Generating response for %NICKNAME%

for %%F in ("%GENERATE_CONFIG_DIR%\*.yaml") do (
    echo Generating response for %%F
    python src\generate_response.py --config_path "%%F"
)

echo Done.
endlocal
