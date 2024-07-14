# One Line script
Please run `chmod +x run.sh && ./run.sh` to reproduce all experiments. This script will do the following:
1. create a conda environment with `python=3.11` and install all necessary packages
2. install the LLM weights (`gpt2`, `Mistral7B`, `Llama3-8B`, and `Gemma7b`) and queries (for datasets `wikitext` and `penn treebank`) and MNL weights and queries (for datasets `MNIST` and `EuroSAT`) from a public Google Drive [link](https://drive.google.com/drive/folders/1aiCuaO9lyvaLyaZMdscgWyx0qz28FbJQ?usp=drive_link)
3. run all adasoftmax experiments and print results 
