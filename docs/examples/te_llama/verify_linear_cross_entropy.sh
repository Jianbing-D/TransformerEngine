#! /bin/bash

# install prerequisites
pip install --upgrade accelerate transformers peft datasets wandb

# run baseline
python main.py