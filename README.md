# Stable Diffusion (SDXL) with Python

## Getting Started

We begin by laying the groundwork in your terminal with the commands below to set up a virtual environment.

'''bash
python3 -m venv .venv
source .venv/bin/activate
'''

and install required libraries

'''bash
pip3 install diffusers --upgrade
pip3 install invisible_watermark transformers accelerate safetensors
pip3 install ipykernel jupyter
'''

## Running

'''
python go.py
'''

## Apple M1

run 'brew install libomp'