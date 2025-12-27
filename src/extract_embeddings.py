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
from utils.embedding_utils import EmbeddingExtractor
import argparse
import logging
import json
import os
import torch

# %%
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# %%
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--prompt_column", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--layers", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


# %%
def extract_embeddings_main(
    model_name: str,
    input_file: str,
    prompt_column: str,
    output_file: str,
    batch_size: int,
    layers=None,
    device: str | None = None,
):
    """
    Programmatic entrypoint for embedding extraction.
    """

    if os.path.exists(output_file):
        logger.info(f"Embeddings already exist in {output_file}, skipping extraction")
        return

    extractor = EmbeddingExtractor(model_name, device=device)

    # -------- Load data --------
    if not (os.path.exists(input_file) and input_file.endswith(".json")):
        raise ValueError(f"Input file {input_file} is not a valid JSON file")

    try:
        with open(input_file, "r") as f:
            data = json.load(f)
        prompts = [item[prompt_column] for item in data]
    except Exception as e:
        raise ValueError(f"Error loading JSON file {input_file}: {e}")

    # -------- Layers --------
    if layers is None:
        layer_ids = list(range(extractor.num_layers))
    elif isinstance(layers, str):
        layer_ids = [int(layer.strip()) for layer in layers.split(",")]
    else:
        layer_ids = list(layers)

    # -------- Extract --------
    embeddings = extractor.extract_embeddings(
        prompts=prompts,
        batch_size=batch_size,
        layers=layer_ids,
    )

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    torch.save(embeddings, output_file)

    logger.info(f"Embeddings saved to {output_file}")


# %%
if __name__ == "__main__":
    args = parse_args()

    extract_embeddings_main(
        model_name=args.model_name,
        input_file=args.input_file,
        prompt_column=args.prompt_column,
        output_file=args.output_file,
        batch_size=args.batch_size,
        layers=args.layers,
        device=args.device
    )
