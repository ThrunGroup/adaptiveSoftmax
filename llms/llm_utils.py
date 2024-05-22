import os
import torch
import numpy as np
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llms.llm_constants import (
    LLM_WEIGHTS_DIR,
    LLM_QUERIES_DIR,

    WIKITEXT_DATASET,
    PENN_TREEBANK_DATASET,
    GPT2,
    LLAMA_3_8B,
    MISTRAL_7B,
    GEMMA_7B,

    MAX_LENGTH,
    NUM_QUERY,
)

def load_llm_data(
        dataset=WIKITEXT_DATASET, 
        model_id=GPT2, 
        num_query=NUM_QUERY,
        testing=False,
    ):
    """
    Reach into the forward function of the model and save the A and xs.
    Will only run if the weights don't already exist
    """
    # loading A (this nver changes per llm)
    model_name = model_id.replace('/', '_')
    os.makedirs(LLM_WEIGHTS_DIR, exist_ok=True)
    weight_path = f"{LLM_WEIGHTS_DIR}/{model_name}.npz"
    if os.path.exists(weight_path):
        A = np.load(weight_path, allow_pickle=False)['weights']
    else:
        A = extract_weights(model_id, weight_path)

    # loading X (this depends on dataset and number of queries)
    os.makedirs(LLM_QUERIES_DIR, exist_ok=True)
    query_path = f"{LLM_QUERIES_DIR}/{model_name}_{dataset}_query{num_query}.npz"
    if os.path.exists(query_path):
        X = np.load(query_path, allow_pickle=False)['queries']
    else:
        X = extract_queries(dataset, model_id, num_query, query_path)

    return A, X


def extract_weights(model_id, save_to):
    """
    Function to extract the A matrix from the LLMs. 
    Returns as a numpy array assigned to cpu 
    """
    model = AutoModelForCausalLM.from_pretrained(model_id)
    if model_id == GPT2:
        A = model.lm_head.weight
    elif model_id in [LLAMA_3_8B, MISTRAL_7B, GEMMA_7B]:
        A = model.lm_head.weight
    
    A = A.detach().cpu().numpy()
    np.savez_compressed(save_to, weights=A)
    return A


def extract_queries(dataset_name, model_id, num_query, save_to):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device is {device}")

    dataset = load_from_datasets(dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    
    # setting the context sizes
    encodings = get_encodings(tokenizer, dataset, dataset_name)
    max_length = get_max_length(model, model_id)
    seq_len = encodings.input_ids.size(1)  
    stride = int((seq_len - max_length) / num_query)
    print("num_queries should be", int((seq_len - max_length)/stride))

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

    np.savez_compressed(save_to, queries=Xs)
    return np.array(Xs)


def get_max_length(model, model_id):
    if model_id == GPT2:
        max_length = model.config.n_positions // 2
    elif model_id in [LLAMA_3_8B, MISTRAL_7B, GEMMA_7B]:
        max_length = model.config.max_position_embeddings // 4
    return min(max_length, MAX_LENGTH)


def get_encodings(tokenizer, dataset, dataset_name):
    # combining texts into single batch
    if dataset_name == WIKITEXT_DATASET:
        key = "text"
    elif dataset_name == PENN_TREEBANK_DATASET:
        key = "sentence"
    else:
        raise NotImplemented("only wikitext and penn treebank supported")
    return tokenizer("\n\n".join(dataset[key]), return_tensors="pt")


def load_from_datasets(
    dataset_name: str = WIKITEXT_DATASET,
    num_samples: int = None,
) -> np.ndarray:
    """
    Given dataset_name, load the dataset with the correct number of samples
    """
    test_set = None
    if dataset_name == WIKITEXT_DATASET:
        test_set = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    elif dataset_name == PENN_TREEBANK_DATASET:
        test_set = load_dataset("ptb_text_only", split="test")

    else:
        # TODO: more datasets that are spiky??
        raise NotImplementedError("Only wikitext supported for now")

    if num_samples is None or num_samples > len(test_set):
            num_samples = len(test_set)
    return test_set[:num_samples]


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
            A, X = load_llm_data(
                dataset=dataset, 
                model_id=model_id, 
                num_query=1000,
                testing=False,
            )
            print(X.shape)