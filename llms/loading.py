"""
Include any loading functions here. 
Examples include loading datasets, models, tokenizers, etc
"""
import numpy as np

from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AutoModelForCausalLM, AutoTokenizer

from llms.llm_constants import (
    WIKITEXT_DATASET,
    GPT2,
    LLAMA_3_8B,
)


def load_from_datasets(
    dataset_name: str = WIKITEXT_DATASET,
    num_samples: int = None,
) -> np.ndarray:
    """
    Given dataset_name, load the dataset with the correct number of samples
    """
    test_set = None
    if dataset_name == WIKITEXT_DATASET:
        print("loading dataset")
        test_set = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        if num_samples is None:
            num_samples = len(test_set)
    else:
        # TODO: more datasets that are spiky??
        raise NotImplementedError("Only wikitext supported for now")

    return test_set[:num_samples]

def load_tokenizer_and_model(model_id=GPT2):
    """
    returns the tokenizer, naive model 
    """
    if model_id == GPT2:
        tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
        model = GPT2LMHeadModel.from_pretrained(model_id)

    elif model_id == LLAMA_3_8B:
        print("loading model")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)

    else:
        raise NotImplementedError("Only GPT2 and Llama supported for now")

    return tokenizer, model

def get_encodings(tokenizer, dataset):
    # combining texts into single batch
    return tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
