#!/bin/bash

conda create -n sftm python=3.11
source activate sftm || { echo "Failed to activate Conda environment"; exit 1; }

# # Upgrade pip and install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# # donload all relevant weights
python -m llms.download_data
python -m mnl.download_data

# # Run the Python module
python main.py
