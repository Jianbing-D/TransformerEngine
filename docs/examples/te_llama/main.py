# Followed by https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/te_llama/tutorial_accelerate_hf_llama_with_te.html#

# This tutorial loads and trains a Llama 3 8B model which takes up most of the GPU memory.

import argparse

# Import necessary packages, methods and variables
from utils import *

def baseline(args):
    # Provide Huggingface Access Token
    hyperparams.hf_access_token = args.HF
    assert hyperparams.hf_access_token, "Provide a HF API Access Token!"
    
    # Set wandb API key if provided
    hyperparams.wandb_api_key = args.wandb_key

    # Provide a directory to cache weights in to avoid downloading them every time.
    # (By default, weights are cached in `~/.cache/huggingface/hub/models`)
    hyperparams.weights_cache_dir = ""

    # For Llama 2, uncomment this line (also set by default)
    # hyperparams.model_name = "meta-llama/Llama-2-7b-hf"

    # For Llama 3, uncomment this line
    hyperparams.model_name = "meta-llama/Meta-Llama-3-8B"

    hyperparams.mixed_precision = "bf16"


    # Init the model and accelerator wrapper
    model = init_baseline_model(hyperparams)
    accelerator, model, optimizer, train_dataloader, lr_scheduler = wrap_with_accelerator(model, hyperparams)


    # Finetune the model
    finetune_model("baseline", model, hyperparams, accelerator, train_dataloader, optimizer, lr_scheduler)

def te_bf16(args):
    """
    Replace HF's LlamaDecoderLayer with TE's TransformerLayer, Precision: BF16
    """
    # Provide Huggingface Access Token
    hyperparams.hf_access_token = args.HF
    assert hyperparams.hf_access_token, "Provide a HF API Access Token!"
    
    # Set wandb API key if provided
    hyperparams.wandb_api_key = args.wandb_key

    # Provide a directory to cache weights in to avoid downloading them every time.
    # (By default, weights are cached in `~/.cache/huggingface/hub/models`)
    hyperparams.weights_cache_dir = ""

    # For Llama 2, uncomment this line (also set by default)
    # hyperparams.model_name = "meta-llama/Llama-2-7b-hf"

    # For Llama 3, uncomment this line
    hyperparams.model_name = "meta-llama/Meta-Llama-3-8B"

    hyperparams.mixed_precision = "bf16"


    # Init the model and accelerator wrapper
    model = init_te_llama_model(hyperparams)
    accelerator, model, optimizer, train_dataloader, lr_scheduler = wrap_with_accelerator(model, hyperparams)


    # Finetune the model
    finetune_model("te_bf16", model, hyperparams, accelerator, train_dataloader, optimizer, lr_scheduler)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--HF", type=str, required=True, help="Huggingface Access Token")
    parser.add_argument("--wandb_key", type=str, help="Weights & Biases API Key")
    parser.add_argument("--which", type=str, required=True, help="Which model to run", 
                        choices=["baseline", "te_bf16"])

    args = parser.parse_args()

    if args.which == "baseline":
        baseline(args)
    elif args.which == "te_bf16":
        te_bf16(args)
    else:
        raise ValueError(f"Invalid model: {args.which}")
