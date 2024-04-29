import os
import torch
import numpy as np

from llms.loading import load_tokenizer_and_model, load_from_datasets, get_encodings
from llms.llm_constants import (
    LLM_WEIGHTS_DIR,
    LLM_XS_DIR,
    WIKITEXT_DATASET,
    GPT2,
    GPT_FINAL_HIDDEN_LAYER_NAME,
    CONTEXT_WINDOW_STRIDE,
    LLAMA_3_8B,
    LLAMA_FINAL_HIDDEN_LAYER_NAME,
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


def register_hook(model, model_id):
    """
    This is just a helper function that sets up the hook with the correct linear layer
    :return: a Callable hook
    """
    # TODO: add more models
    if model_id == GPT2:
        layer_name = GPT_FINAL_HIDDEN_LAYER_NAME
    elif model_id == LLAMA_3_8B:
        layer_name = LLAMA_FINAL_HIDDEN_LAYER_NAME
    else:
        raise NotImplementedError("only supports gpt2 and llama for now")

    # set up hook
    layer_to_hook = getattr(model, layer_name)
    return layer_to_hook.register_forward_hook(extract_final_hidden_state)
    

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
    # we only want the final hidden state. TODO: is there cleaner way to do this?
    final_hidden_state = output[0] if isinstance(output, tuple) else output