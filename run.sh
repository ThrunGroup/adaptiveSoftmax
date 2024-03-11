#!/bin/bash

# Upgrade pip and install dependencies
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# Run the Python module
python3 -m llms.gpt2.gpt_experiments
