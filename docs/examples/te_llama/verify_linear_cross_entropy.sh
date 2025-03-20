#! /bin/bash

# install prerequisites
pip install -r requirements.txt

# run baseline
HF=xxxx # huggingface token to access llama3 8B model weights
wandb_key=yyyy # wandb api key

python main.py --HF ${HF} --wandb_key ${wandb_key} --which=baseline
python main.py --HF ${HF} --wandb_key ${wandb_key} --which=te_bf16
python main.py --HF ${HF} --wandb_key ${wandb_key} --which=te_bf16_linear_cross_entropy