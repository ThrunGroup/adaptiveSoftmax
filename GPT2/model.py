import torch
from transformers import (
    GPT2Config,
    GPT2ForSequenceClassification,      # CBT falls in here
    GPT2LMHeadModel,
    AutoTokenizer,
)

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


def load_model(
    model_name: str,
    is_lmtask: bool,
):
    """
    Load a pretrained GPT2 model from HuggingFace and apply AdaSoftmax
    -> model_name is one of [gpt2, gpt2-medium, gpt2-large, gpt2-xl]
    """
    # loading GPT2 model
    model_type = GPT2LMHeadModel if is_lmtask else GPT2ForSequenceClassification

    config = GPT2Config.from_pretrained(model_name)
    model = model_type.from_pretrained(model_name, config=config)
    model.to(device)
    model.config.pad_token_id = model.config.eos_token_id

    tokenizer = AutoTokenizer



