import torch
import torch.nn.functional as F
import numpy as np

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset
from copy import deepcopy
from typing import Union, Tuple, Optional, Callable, List
from tqdm import tqdm

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'adaptive_softmax'))
from adasoftmax import ada_softmax, precompute_mu
from gpt_constants import (
    MULTIPLICATIVE_ERROR,
    DELTA_ERROR,
    WIKITEXT_BETA,
    PRECOMPUTE,
)


def load_from_datasets(
    dataset_name: str = "wikitext",
    num_samples: int = None,
):
    """
    Given dataset_name, load the dataset with the correct number of samples
    """
    test_set = None
    if dataset_name == "wikitext":
        test_set = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        if num_samples is None:
            num_samples = len(test_set)
    else:
        print("doesn't support any other datasets as of now")

    return test_set[:num_samples]


def check_correctness(
    naive_results: torch.Tensor,
    adaptive_results: torch.Tensor,
    verbose: bool = False,
) -> bool:
    """
    Function that checks if the two results are within epsilon multiplicative error (1 - delta)% of the time.
    :param naive_results: the logits of the naive softmax
    :param adaptive_results: the results of adasoftmax
    :returns: whether we are within theoretical gaurantee defined above
    """
    not_within = 0
    num_exp = len(naive_results)

    for i in range(num_exp):
        naive_logit = naive_results[i].item()
        ada_logit = adaptive_results[i].item()
        if abs(naive_logit - ada_logit) > MULTIPLICATIVE_ERROR * naive_logit:
            not_within += 1

    if verbose:
        print(f"=> delta error is {not_within / num_exp} for epsilon = {MULTIPLICATIVE_ERROR}" )

    return (not_within / num_exp) < DELTA_ERROR


def get_gains(
    true_mu: np.ndarray,
    naive_budget: int,
    adaptive_budget: int,
    verbose: bool = False,
) -> float:
    """
    Given both naive and adaptive budget, find the proportional gains.
    :param true_mu: used to get the upper bound on gain (this is a loose proxy)
    :param naive_budget: the naive budget
    :param adaptive_budget: the adaptive budget
    :param verbose: whether you print theoretical gain
    """
    empirical_gain = naive_budget / adaptive_budget
    if verbose:
        n_classes = true_mu.shape[0]
        l2 = np.sum(np.exp(2 * (true_mu - np.max(true_mu))))
        l1 = np.sum(np.exp(true_mu - np.max(true_mu))) ** 2
        theoretical_gain = n_classes * l2 / l1

        print("=> Theoretical gain: ", theoretical_gain)
        print("=> Empirical gain: ", empirical_gain)

    return empirical_gain


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
        verbose: bool = False,
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
        flattened_states = hidden_states.view(-1, hidden_states.size(-1))   # TODO: currently only supports batch = 1

        A = model.lm_head.weight.data.numpy()   # size = (embed, vocab_size)
        x = flattened_states[-1].numpy()

        best_arms, z, adaptive_budget = ada_softmax(
            A=A,
            x=x,
            # samples_for_sigma=flattened_states.shape[0],
            samples_for_sigma=None,  # this finds the exact sigma <- debugging purposes
            beta=WIKITEXT_BETA,   # mu is very spiky
            verbose=verbose,
        )
        likelihood = torch.tensor(z)

        # FOR DEBUGGING PURPOSES. ERASE WHEN FINISHED
        print("best arms from adasoftmax: ", best_arms)
        print("\ngroundtruth best arms: ", np.argmax(A@x))

        return likelihood, adaptive_budget

    return adaptive_forward


def run_experiment(
    exp_type: str = "both",
    dataset_name: str = "wikitext",
    num_samples: int = None,
    model_id: str = "gpt2",
    stride: int = 512,
    verbose: bool = True,
):
    """
    This function runs the given exp_type on the GPT model.
    :param: either "correctness", "gains", or "both"
    :param dataset_name: the dataset we're testing on
    :param num_samples: the number of samples for the dataset
    :param model_id: the model id of the gpt2 series to run experiment on
    """
    # TODO: currently only works on batch_size = 1
    # TODO: why is device on cpu even on the cluster??
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    naive_model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    naive_shape = naive_model.lm_head.weight.shape  # tensor with 2 elems

    # replace the forward function with the adaptive forward
    adaptive_model = deepcopy(naive_model)
    adaptive_model.forward = get_adaptive_forward(naive_model)

    test = load_from_datasets(dataset_name, num_samples)
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

        with torch.no_grad():
            naive_logits = naive_model(input_ids, labels=None, return_dict=False)[0]
            flattened_naive_logits = naive_logits.view(-1, naive_logits.size(-1))
            naive_ll = F.softmax(WIKITEXT_BETA * flattened_naive_logits, dim=1)[-1, target_id]  # TODO: should be (batch_size, 1)
            naive_budget = naive_shape[0] * naive_shape[1]

            adaptive_ll, adaptive_budget = adaptive_model(
                input_ids,
                labels=None,
                verbose=verbose
            )

        # CELoss averages the losses. But, we're only comparing likelihood
        naive_lls.append(naive_ll)
        adaptive_lls.append(adaptive_ll[target_id])

        if end_loc == seq_len:
            break

    is_correct, gain = None, None
    if exp_type == "both" or exp_type == "correctness":
        is_correct = check_correctness(naive_lls, adaptive_lls, verbose=True)
    if exp_type == "both" or exp_type == "gains":
        gain = get_gains(flattened_naive_logits.numpy(), naive_budget, adaptive_budget, verbose=True)

    print(f"==> Experiment {exp_type} is {is_correct} with gain {gain}")


if __name__ == "__main__":
    run_experiment(num_samples=100, verbose=False)
