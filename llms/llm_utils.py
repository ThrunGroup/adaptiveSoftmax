import os
import torch
import numpy as np

from llms.loading import load_tokenizer_and_model, load_from_datasets, get_encodings
from llms.llm_constants import (
    LLM_WEIGHTS_DIR,
    LLM_XS_DIR,
    WIKITEXT_DATASET,
    GPT2,
    CONTEXT_WINDOW_STRIDE,
)

def load_llm_matrices(
        dataset=WIKITEXT_DATASET, 
        model_id=GPT2, 
        stride=CONTEXT_WINDOW_STRIDE,
        testing=True,
    ):
    """
    Reach into the forward function of the model and save the A and xs.
    Will only run if the weights don't already exist
    """
    os.makedirs(LLM_WEIGHTS_DIR, exist_ok=True)
    os.makedirs(LLM_XS_DIR, exist_ok=True)
    path = f"{model_id}_{dataset}_{stride}.npz"
    if testing:
        path = f"testing_{path}"

    weights_path = f'{LLM_WEIGHTS_DIR}/{path}'
    x_matrix_path = f'{LLM_XS_DIR}/{path}'

    # Check if the files exist
    if os.path.exists(weights_path) and os.path.exists(x_matrix_path):
        A = np.load(weights_path, allow_pickle=False)['data']
        x_matrix = np.load(x_matrix_path, allow_pickle=False)['data']
    else:
        A, x_matrix = get_llm_matrices(dataset, model_id, stride)
        np.savez_compressed(weights_path.rstrip('.npz'), data=A)
        np.savez_compressed(x_matrix_path.rstrip('.npz'), data=x_matrix) 
    return A, x_matrix


def get_llm_matrices(dataset, model_id, stride):
    """
    Run the forward function and retrieve the A and xs.
    """
    # TODO: check dimensions!
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device is {device}")

    dataset = load_from_datasets(dataset)
    tokenizer, model = load_tokenizer_and_model(model_id)
    model = model.to(device)
    A = model.lm_head.weight.data.cpu().numpy()  # stays the same per stride
    
    # setting the context sizes
    encodings = get_encodings(tokenizer, dataset)
    max_length = model.config.n_positions
    seq_len = encodings.input_ids.size(1)  

    Xs = []
    for begin in (range(0, seq_len, stride)):
        end = min(begin + max_length, seq_len)
        tokens = encodings.input_ids[:, begin:end].to(device)
        input_ids = tokens[:, :-1].contiguous()  # target_id would be tokens[:, -1]

        hook = model.transformer.register_forward_hook(extract_final_hidden_state)
        with torch.no_grad():
            # this populates final_hidden_state variable 
            # NOTE: logits is unused but is left the same variable name for clarity
            logits = model(input_ids)  
            
            sequence_outputs = final_hidden_state.view(-1, final_hidden_state.size(-1))
            x = sequence_outputs[-1].cpu().numpy()  
            Xs.append(x)
        hook.remove()

    return A, Xs
    

def extract_final_hidden_state(module, input, output):
    """ 
    We will hook this function to the model's forward pass to extract the hidden states. 
    :param module: the layer that will triger the hook
    :param input: the input tensors to the module
    :param output: output of module

    Example:
        hook = model.<layer name>.register_forward_hook(extract_final_hidden_state)
        final_output = model.forward(...)
        <do something with final_hidden_state variable>
        hook.remove()
    """
    global final_hidden_state
    #hidden_states = output
    final_hidden_state = output[0] 