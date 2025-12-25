from transformers import LlamaConfig, Qwen2Config, Gemma2Config
from transformers import LlamaForCausalLM, Qwen2ForCausalLM, Gemma2ForCausalLM
from AlphaSteerModel import *
from NaiveSteerModel import *

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


__all__ = [
    "MODELS_DICT", "AlphaSteer_MODELS_DICT", "Steer_MODELS_DICT",
    "AlphaSteer_STEERING_LAYERS", "AlphaSteer_CALCULATION_CONFIG",
]

MODELS_DICT = {
    "qwen2.5": (Qwen2ForCausalLM, Qwen2Config, "Qwen/Qwen2.5-7B-Instruct"),
}

AlphaSteer_MODELS_DICT = {
    "qwen2.5": (AlphaQwen2ForCausalLM, Qwen2Config, "Qwen/Qwen2.5-7B-Instruct"),
}

Steer_MODELS_DICT = {
    "qwen2.5": (SteerQwen2ForCausalLM, Qwen2Config, "Qwen/Qwen2.5-7B-Instruct"),
}

AlphaSteer_STEERING_LAYERS = {
    "qwen2.5": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20],
}

AlphaSteer_CALCULATION_CONFIG = {
    "qwen2.5": [(5, 0.6), (6, 0.6), (7, 0.6), (8, 0.6), (9, 0.5), (10, 0.6), (11, 0.5), (12, 0.5), (13, 0.5), (14, 0.3), (15, 0.3), (16, 0.5), (18, 0.5), (19, 0.6)], 
}