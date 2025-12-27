# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
# ---

# %%
import os
import glob
import subprocess
from pathlib import Path
from extract_embeddings import extract_embeddings_main
from calc_steering_matrix import calc_steering_main

# %% [markdown]
# ======================
# Configuration
# ======================

# %%
TRAIN_VAL_DIR = "data/instructions/train_val"
EMBEDDING_DIR = "data/embeddings/TinyLlama"
NICKNAME = "TinyLlama"
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cpu"

# %%
STEERING_SAVE_PATH = f"data/steering_matrix/steering_matrix_{NICKNAME}.pt"
GENERATE_CONFIG_DIR = "config/TinyLlama"

# %%
# Ensure output directories exist
os.makedirs(EMBEDDING_DIR, exist_ok=True)
os.makedirs(os.path.dirname(STEERING_SAVE_PATH), exist_ok=True)

# %% [markdown]
# ======================
# Extract embeddings
# ======================

# %%
json_files = glob.glob(os.path.join(TRAIN_VAL_DIR, "*.json"))

# %%
for file_path in json_files:
    filename = Path(file_path).stem
    print(f"Extracting embeddings for {file_path}")

    # Match bash logic
    if "coconot" in filename:
        prompt_column = "prompt"
    else:
        prompt_column = "query"

    output_file = os.path.join(EMBEDDING_DIR, f"embeds_{filename}.pt")
    extract_embeddings_main(
        model_name=MODEL_NAME,
        input_file=file_path,
        prompt_column=prompt_column,
        output_file=output_file,
        batch_size=16,
        device=DEVICE
    )
# %% [markdown]
# ======================
# Calculate steering matrix
# ======================
# %%
print(f"Calculating steering matrix for {NICKNAME}")
STEERING_SAVE_PATH="data/steering_matrix/steering_matrix_${NICKNAME}.pt"

calc_steering_main(
    embedding_dir=EMBEDDING_DIR,
    model_name=MODEL_NAME,
    save_path=STEERING_SAVE_PATH,
    device=DEVICE
)

# %% [markdown]
# # ======================
# # Generate responses
# # ======================

# %%
print(f"Generating responses for {NICKNAME}")
yaml_files = glob.glob(os.path.join(GENERATE_CONFIG_DIR, "*.yaml"))
for yaml_path in yaml_files:
    print(f"Generating response for {yaml_path}")

# %% [markdown]
#     subprocess.run(
#         [
#             "python",
#             "src/generate_response.py",
#             "--config_path", yaml_path,
#         ],
#         check=True,
#     )
