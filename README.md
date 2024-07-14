# One Line script
Please run `chmod +x run.sh && ./run.sh` to generate reproduce all experiments. This script will do the following:
1. create a conda environment with `python=3.11`
2. install the necessary LLM weights (`gpt2`, `Mistral7B`, `Llama3-8B`, and `Gemma7b`) and queries (datasets `wikitext` and `penn treebank`) from a public Google Drive [link](https://drive.google.com/drive/folders/1aiCuaO9lyvaLyaZMdscgWyx0qz28FbJQ?usp=drive_link)
3. run all adasoftmax experiments and print results 
