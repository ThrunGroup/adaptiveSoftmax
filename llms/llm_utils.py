import os
import torch
import numpy as np
from tqdm import tqdm

from llms.loading import load_tokenizer_and_model, load_from_datasets, get_encodings
from llms.llm_constants import (
    LLM_WEIGHTS_AND_XS_DIR,
    WIKITEXT_DATASET,
    PENN_TREEBANK_DATASET,
    GPT2,
    LLAMA_3_8B,
    MISTRAL_7B,
    GEMMA_7B,

    MAX_LENGTH,
    NUM_QUERY
)

def load_llm_matrices(
        dataset=WIKITEXT_DATASET, 
        model_id=GPT2, 
        num_query=NUM_QUERY,
        testing=False,
    ):
    """
    Reach into the forward function of the model and save the A and xs.
    Will only run if the weights don't already exist
    """
    os.makedirs(LLM_WEIGHTS_AND_XS_DIR, exist_ok=True)
    path = f"{model_id}_{dataset}_query{num_query}.npz".replace('/', '_')
    if testing: path = f"testing_{path}"
    full_path = f'{LLM_WEIGHTS_AND_XS_DIR}/{path}'

    # Load if file exists. Otherwise generate and save
    if os.path.exists(full_path):
        A, X = np.load(full_path, allow_pickle=False).values()
    else:
        A, X = get_llm_matrices(dataset, model_id, num_query)
        np.savez_compressed(full_path, A=A, X=X)

    return A, X


def get_llm_matrices(datase_name, model_id, num_query):
    """
    Run the forward function and retrieve the A and xs.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device is {device}")

    dataset = load_from_datasets(datase_name)
    tokenizer, model = load_tokenizer_and_model(model_id)
    model = model.to(device)
    A = extract_A(model, model_id)
    
    # setting the context sizes
    encodings = get_encodings(tokenizer, dataset, datase_name)
    max_length = get_max_length(model, model_id)
    seq_len = encodings.input_ids.size(1)  
    stride = int((seq_len - max_length) / num_query)

    Xs = []
    for begin in tqdm(range(0, seq_len, stride)):
        end = min(begin + max_length, seq_len)
        tokens = encodings.input_ids[:, begin:end].to(device)
        input_ids = tokens[:, :-1].contiguous()  # target_id would be tokens[:, -1]

        hook = register_hook(model, model_id)
        with torch.no_grad():
            # this populates final_hidden_state variable 
            # NOTE: logits is unused but is left the same variable name for clarity
            logits = model(input_ids)  

            sequence_outputs = final_hidden_state.view(-1, final_hidden_state.size(-1))
            x = sequence_outputs[-1].cpu().numpy()  
            Xs.append(x)
        hook.remove()

    return A, Xs


def get_max_length(model, model_id):
    if model_id == GPT2:
        max_length = model.config.n_positions // 2
    elif model_id in [LLAMA_3_8B, MISTRAL_7B, GEMMA_7B]:
        max_length = model.config.max_position_embeddings // 4

    return min(max_length, MAX_LENGTH)


def extract_A(model, model_id):
    """
    Function to extract the A matrix from the LLMs. 
    Returns as a numpy array assigned to cpu 
    """
    if model_id == GPT2:
        A = model.lm_head.weight
    elif model_id in [LLAMA_3_8B, MISTRAL_7B, GEMMA_7B]:
        A = model.lm_head.weight
    return A.detach().cpu().numpy()


def register_hook(model, model_id):
    """
    This is just a helper function that sets up the hook with the correct linear layer
    :return: a Callable hook
    """
    # TODO: add more models
    if model_id == GPT2:
        layer_to_hook = model.transformer.ln_f 

    elif model_id in [LLAMA_3_8B, MISTRAL_7B, GEMMA_7B]:
        layer_to_hook = model.model.norm # hidden outputs get normalized (dims are the same)

    else:
        # TODO: add more models
        raise NotImplementedError("only supports gpt2 and llama for now")

    # set up hook
    return layer_to_hook.register_forward_hook(extract_final_hidden_state)
    

def extract_final_hidden_state(module, input, output):
    """ 
    We will hook this function to the model's forward pass to extract the hidden states.
    The final hidden states correspond to the xs 
    :param module: the layer that will trigger the hook
    :param input: the input tensors to the module
    :param output: output of module

    Example:
        hook = model.<layer name>.register_forward_hook(extract_final_hidden_state)
        final_output = model.forward(...)
        <do something with final_hidden_state variable>
        hook.remove()
    """
    global final_hidden_state
    final_hidden_state = output


if __name__ == "__main__":
    for dataset in [WIKITEXT_DATASET, PENN_TREEBANK_DATASET]:
        print(f"=> dataset {dataset}")
        for model_id in [GPT2, LLAMA_3_8B, MISTRAL_7B, GEMMA_7B]:
            print(f"=> model_id {model_id}")
            load_llm_matrices(
                dataset=dataset, 
                model_id=model_id, 
                num_query=1000,
                testing=False,
            )