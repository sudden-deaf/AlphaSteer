@echo off
setlocal EnableDelayedExpansion

REM =========================
REM Configuration
REM =========================

set TRAIN_VAL_DIR=data\instructions\train_val
set EMBEDDING_DIR=data\embeddings\llama3.1
set NICKNAME=llama3.1
set MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
set DEVICE=cuda:0

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

echo Done.
endlocal
