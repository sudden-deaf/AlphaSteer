@echo off
setlocal

REM =========================
REM Configuration
REM =========================

set TRAIN_VAL_DIR=data\instructions\train_val
set EMBEDDING_DIR=data\embeddings\qwen2.5
set NICKNAME=qwen2.5
set MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
set DEVICE=cuda:0

REM =========================
REM Steering / Generation
REM =========================

set GENERATE_CONFIG_DIR=config\qwen2.5
echo Generating response for %NICKNAME%

for %%F in ("%GENERATE_CONFIG_DIR%\*.yaml") do (
    echo Generating response for %%F
    python src\generate_response.py --config_path "%%F"
)

echo Done.
endlocal
