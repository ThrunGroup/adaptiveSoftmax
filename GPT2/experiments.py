import torch
import torch.nn.functional as F
import numpy as np

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset
from copy import deepcopy
from typing import Union, Tuple, Optional, Callable
from tqdm import tqdm

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'adaptive_softmax'))
from adasoftmax import ada_softmax
from gpt_constants import (
    MULTIPLICATIVE_ERROR,
    DELTA_ERROR,
)


def get_adaptive_forward(model) -> Callable:
    """
    Function that redefines the forward pass for GPT2. We do this by switching out
    lm_head + CELoss to log(adaSoftmax) + NLLLoss. The input parameters are the same throughout and the forward
    pass should return the log probabilities instead of a dictionary

    :param model: the GPT2LMHeadModel
    :param: the rest are the same as the original model
    :returns: the forward pass of the model
    """
    def adaptive_forward(
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
    ) -> torch.Tensor:
        labels = None   # we're only concerned with the log likelihood
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
        # converting (batch, sequence, embed) -> (batch x sequence, embed)
        hidden_states = transformer_outputs[0]
        flattened_states = hidden_states.view(-1, hidden_states.size(-1))

        adaptive_results = []
        A = model.lm_head.weight.data.numpy()   # size = (embed, vocab_size)
        _, z, _ = ada_softmax(A=A, x=flattened_states[-1].numpy(), samples_for_sigma=flattened_states.shape[0])
        # for state in flattened_states:
        #     _, z, _ = ada_softmax(A=A, x=state.numpy(), samples_for_sigma=state.shape[0])
        #     adaptive_results.append(z)
        # adaptive_results_np = np.array(adaptive_results)
        # likelihood = torch.tensor(adaptive_results_np)
        likelihood = torch.tensor(z)
        return likelihood

    return adaptive_forward


def sanity_check():
    # TODO: why is device on cpu even on the cluster??
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "gpt2"
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    naive_model = GPT2LMHeadModel.from_pretrained(model_id).to(device)

    # replace the forward function with the adaptive forward
    adaptive_model = deepcopy(naive_model)
    adaptive_model.forward = get_adaptive_forward(naive_model)

    num_samples = 100
    stride = 512
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")[:num_samples]
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    max_length = naive_model.config.n_positions
    seq_len = encodings.input_ids.size(1)

    naive_lls = []
    adaptive_lls = []

    # context size is fixed at seq_len. The context gets shifted by stride amount per experiment
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        tokens = encodings.input_ids[:, begin_loc:end_loc].to(device)
        input_ids = tokens[:, :-1].contiguous()
        target_id = tokens[:, -1]

        # # get log likelihoods for the target only
        # with torch.no_grad():
        #     naive_logits = naive_model(input_ids, labels=None, return_dict=False)[0]
        #     flattened_naive_logits = naive_logits.view(-1, naive_logits.size(-1))
        #     import ipdb; ipdb.set_trace()
        #     naive_ll = F.softmax(flattened_naive_logits, dim=1)  # TODO: should be (batch_size, 1)
        #     adaptive_ll = adaptive_model(input_ids, labels=None)
        #
        # # CELoss averages the losses.
        # naive_lls.append(torch.mean(naive_ll[:, target_id], dim=0))
        # adaptive_lls.append(torch.mean(adaptive_ll[:, target_id], dim=0))
        # get log likelihoods for the target only
        with torch.no_grad():
            naive_logits = naive_model(input_ids, labels=None, return_dict=False)[0]
            flattened_naive_logits = naive_logits.view(-1, naive_logits.size(-1))
            naive_ll = F.softmax(flattened_naive_logits, dim=1)[-1, target_id]  # TODO: should be (batch_size, 1)
            adaptive_ll = adaptive_model(input_ids, labels=None)[target_id]

        # CELoss averages the losses. But, we're only comparing likelihood
        naive_lls.append(naive_ll)
        adaptive_lls.append(adaptive_ll)

        if end_loc == seq_len:
            break

    # check to see if the two results are within epsilon multiplicative error (1 - delta)% of the time.
    not_within = 0
    num_exp = len(adaptive_lls)
    for i in range(num_exp):
        ada_logit = adaptive_lls[i].item()
        naive_logit = naive_lls[i].item()
        if abs(ada_logit - naive_logit) > MULTIPLICATIVE_ERROR * naive_logit:
            not_within += 1

    print(f"delta error is {not_within / num_exp}")
    return (not_within / num_exp) < DELTA_ERROR


if __name__ == "__main__":
    sanity_check()
