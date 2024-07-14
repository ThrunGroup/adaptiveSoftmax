import gdown
import os
from llms.llm_constants import (
    GOOGLE_DRIVE_PREFIX,

    WIKITEXT_DATASET,
    PENN_TREEBANK_DATASET,

    LLM_WEIGHTS_DIR,
    GPT2,
    LLAMA_3_8B,
    MISTRAL_7B,
    GEMMA_7B,

    LLM_QUERIES_DIR,

)

def download(files_info, dest):
    os.makedirs(dest, exist_ok=True)
    for link, name in files_info: 
        npz_file = f"{name.replace('/', '_')}.npz"
        output_path = os.path.join(dest, npz_file)  
        gdown.download(link, output_path, quiet=False)

def download_weights():
    weight_files = [
        (f'{GOOGLE_DRIVE_PREFIX}16ANScS-tPpHO20n4ygJQiXXFt0Z6utzs', GPT2),
        (f'{GOOGLE_DRIVE_PREFIX}16B1c_7ZTq5AgfFD1_71sAVuZB7bwK51R', LLAMA_3_8B),
        (f'{GOOGLE_DRIVE_PREFIX}1652IC2KVbNmHX17ceQEZ6xbKla7vn-_H', GEMMA_7B),
        (f'{GOOGLE_DRIVE_PREFIX}16DPk8NCxhrEP8X6hd7g7kruGAUSUaB8i', MISTRAL_7B)
    ]
    download(weight_files, LLM_WEIGHTS_DIR)

def download_queries():
    query_files = [
        (f'{GOOGLE_DRIVE_PREFIX}15vrp2aFXkUXO2bkXpP4dU4JusoGFzrTJ', f"{GPT2}_{WIKITEXT_DATASET}_query{1000}"),
        (f'{GOOGLE_DRIVE_PREFIX}15wZdj7KVRek-_-FXTZEtRHeyL7dioCXr', f"{GPT2}_{PENN_TREEBANK_DATASET}_query{1000}"),
        (f'{GOOGLE_DRIVE_PREFIX}163GeL1F55A3rjITyrmHO24LN_rYTLCTr', f"{LLAMA_3_8B}_{WIKITEXT_DATASET}_query{1000}"),
        (f'{GOOGLE_DRIVE_PREFIX}15vJ6azuUK80cA9gCfiXbbF-WNhwue2W-', f"{LLAMA_3_8B}_{PENN_TREEBANK_DATASET}_query{1000}"),
        (f'{GOOGLE_DRIVE_PREFIX}15jeH0Ga6EXHD3vp-qA9ZSxcEJJ74MZKE', f"{GEMMA_7B}_{WIKITEXT_DATASET}_query{1000}"),
        (f'{GOOGLE_DRIVE_PREFIX}15mTmYMnewWYRJZsob7beGAjleM5KH8u-', f"{GEMMA_7B}_{PENN_TREEBANK_DATASET}_query{1000}"),
        (f'{GOOGLE_DRIVE_PREFIX}164CSbrs_PhMhmw4EJ7U64XlAeUG2grJB', f"{MISTRAL_7B}_{WIKITEXT_DATASET}_query{1000}"),
        (f'{GOOGLE_DRIVE_PREFIX}161HaM9-K579-v8YjW1PjiP3xZcqv4tA6', f"{MISTRAL_7B}_{PENN_TREEBANK_DATASET}_query{1000}") 
    ]
    download(query_files, LLM_QUERIES_DIR)

if __name__ == "__main__":
    download_weights()
    download_queries()


