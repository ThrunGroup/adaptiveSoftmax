"""
Include any loading functions here. 
Examples include loading datasets, models, tokenizers, etc
"""
import numpy as np

from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel, 
    GPT2TokenizerFast,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    LlamaForCausalLM,
    MistralForCausalLM,
    AutoModelForCausalLM,
 ) 

from llms.llm_constants import (
    WIKITEXT_DATASET,
    PENN_TREEBANK_DATASET,
    GPT2,
    LLAMA_3_8B,
    MISTRAL_7B,
    GEMMA_7B,
    NUM_QUERY,
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
        test_set = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    elif dataset_name == PENN_TREEBANK_DATASET:
        test_set = load_dataset("ptb_text_only", split="test")

    else:
        # TODO: more datasets that are spiky??
        raise NotImplementedError("Only wikitext supported for now")

    if num_samples is None or num_samples > len(test_set):
            num_samples = len(test_set)
    return test_set[:num_samples]


def load_tokenizer_and_model(model_id=GPT2):
    """
    returns the tokenizer, naive model 
    """
    if model_id == GPT2:
        tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
        model = GPT2LMHeadModel.from_pretrained(model_id)

    elif model_id == LLAMA_3_8B:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id)
        model = LlamaForCausalLM.from_pretrained(model_id)

    elif model_id == MISTRAL_7B:
        tokenizer = AutoTokenizer.from_pretrained(model_id)  # TODO: which one does mistral use?
        model = MistralForCausalLM.from_pretrained(model_id)

    elif model_id == GEMMA_7B:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)

    else:
        raise NotImplementedError("Only GPT2 and Llama supported for now")

    return tokenizer, model

def get_encodings(tokenizer, dataset, dataset_name):
    # combining texts into single batch
    if dataset_name == WIKITEXT_DATASET:
        key = "text"
    elif dataset_name == PENN_TREEBANK_DATASET:
        key = "sentence"
    else:
        raise NotImplemented("only wikitext and penn treebank supported")

    return tokenizer("\n\n".join(dataset[key]), return_tensors="pt")


if __name__ == "__main__":
    for dataset in [WIKITEXT_DATASET, PENN_TREEBANK_DATASET]:
        for model_id in [GPT2, LLAMA_3_8B, MISTRAL_7B, GEMMA_7B]:
            load_llm_matrices(
                dataset=dataset, 
                model_id=model_id, 
                num_query=1000,
                testing=False,
            )

