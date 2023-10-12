import torch
import torch.nn as nn
import os, sys
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset
from typing import Union, Tuple
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'adaptive_softmax'))
from adasoftmax import ada_softmax
from constants import *


def get_adaptive_forward(
        model,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:

    """
    Function that redefines the forward pass for GPT2. We do this by switching out
    lm_head + CELoss to log(adaSoftmax) + NLLLoss. The input parameters are the same throughout and the forward
    pass should return the log probabilities instead of a dictionary

    :param model: the GPT2LMHeadModel
    :param: the rest are the same as the original model
    :returns: the forward pass of the model
    """
    labels = None   # NOTE: we only care about the probability so labels isn't used.
    transformer_outputs = model.transformer(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    hidden_states = transformer_outputs[0]

    # TODO(@lukehan, ryank): this is inefficient. Add support for tensors and collection of x
    adaptive_results = []
    for h_state in hidden_states:
        _, z, _ = ada_softmax(
            A=model.lm_head.numpy(),
            x=h_state.numpy(),
        )
        adaptive_results.append(z)

    log_probabilities = torch.log(torch.tensor(adaptive_results))
    return log_probabilities


def sanity_check():
    device = "gpu" if torch.cuda.is_available else "cpu"
    model_id = "gpt2"
    naive_model = GPT2LMHeadModel.from_pretrained(model_id).to(device)

    # replace the forward function with the adaptive forward
    adaptive_model = naive_model.copy()
    adaptive_model.forward = get_adaptive_forward(naive_model)

    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    max_length = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)

    naive_lls = []
    adaptive_lls = []

    # get the log likelihoods of the next word for naive and adaptive models
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = None   # don't need targets if we're just checking log likelihood

        with torch.no_grad():
            # this will return (lm_logits,) + transformer_outputs[1:]
            naive_outputs = naive_model(input_ids, labels=target_ids, return_dict=False)
            naive_ll = nn.LogSoftmax(naive_outputs[0])  # TODO(@ryank): does this indexing behave as intended?
            adaptive_ll = adaptive_model(input_ids, labels=target_ids)

        naive_lls.append(naive_ll)
        adaptive_lls.append(adaptive_ll)

        if end_loc == seq_len:
            break

    # check to see if the two results are within epsilon multiplicative error
    is_within = 0
    num_exp = len(adaptive_lls)
    for i in range(num_exp):
        if adaptive_lls[i] < (1 + MULTIPLICATIVE_ERROR) * naive_lls[i]:
            is_within += 1

    return (is_within / num_exp) < DELTA_ERROR


if __name__ == "__main__":
    sanity_check()
