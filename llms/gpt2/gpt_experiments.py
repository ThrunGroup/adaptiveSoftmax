import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset
import copy
from typing import Union, Tuple, Optional, Callable, List
from tqdm import tqdm

sys.path.append("/content/drive/MyDrive")
from adaptive_softmax.adasoftmax import ada_softmax
from gpt_constants import (
    MULTIPLICATIVE_ERROR,
    DELTA_ERROR,
    WIKITEXT_BETA,
)


def load_from_datasets(
    dataset_name: str = "wikitext",
    num_samples: int = None,
) -> np.ndarray:
    """
    Given dataset_name, load the dataset with the correct number of samples
    """
    test_set = None
    if dataset_name == "wikitext":
        test_set = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        if num_samples is None:
            num_samples = len(test_set)
    else:
        print("We only support wikitext for now")

    return test_set[:num_samples]


def check_correctness(
    naive_results: torch.Tensor,
    adaptive_results: torch.Tensor,
        eps: float = MULTIPLICATIVE_ERROR,
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
        if abs(naive_logit - ada_logit) > eps * naive_logit:
            not_within += 1

    if verbose:
        print(
            f"=> delta error is {not_within / num_exp} for epsilon = {eps}"
        )

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

        # TODO(@ryank): Explain where this computation comes from. What eqn in the paper?
        l2 = np.sum(np.exp(2 * (true_mu - np.max(true_mu))))
        l1 = np.sum(np.exp(true_mu - np.max(true_mu))) ** 2
        theoretical_gain = n_classes * l2 / l1

        print("=> Theoretical gain: ", theoretical_gain)
        print("=> Empirical gain: ", empirical_gain)

    return empirical_gain

def debug(
        A,
        x,
):
    # Question 1: Visualize variance
    data = A @ x
    subset = np.random.choice(data, size=100, replace=False)
    plt.scatter(range(len(subset)), subset)
    plt.title('mu values')
    plt.savefig('arms.png')

    # Question 2: Investigate variance reduction from importance sampling
    num_samples = 1000
    num_rows = A.shape[0]
    d = A.shape[1]

    unif_var = np.zeros(num_rows)
    uni_dist = np.ones(d)/d
    importance_var = np.zeros(num_rows)
    imp_dist = np.abs(x)/np.sum(np.abs(x))
    var_array = []

    for unif_mix_factor in np.linspace(0, 1, num=10):
      for i in range(num_rows):
        a = A[i]

        # Uniform sampling
        coord = np.random.choice(d, p=uni_dist, size=num_samples)
        samples = a[coord]*x[coord]/uni_dist[coord]
        unif_var[i] = np.var(samples)

        # Importance sampling
        prob_dist = (1-unif_mix_factor) * imp_dist + unif_mix_factor * uni_dist
        prob_dist = prob_dist/prob_dist.sum()  # numerical stability issues before
        coord = np.random.choice(d, p=prob_dist, size=num_samples)
        samples = a[coord] * x[coord] / prob_dist[coord]
        importance_var[i] = np.var(samples)
        mean_var = np.mean(importance_var)
        var_var = np.var(importance_var)

      var_array.append(var_var)
      print("factor: ", unif_mix_factor)
      print("mean variance: ", mean_var)
      print("variance of variance: ", var_var)

    plt.scatter(np.linspace(0, 1, num=10), var_array)
    plt.title("unif_mix_factor to variance")
    plt.savefig("unif_mix_factor to variance.png")

    print("mean (uniVar, impVar)", (np.mean(unif_var),np.mean(importance_var)))
    print("variance of the variances", np.var(importance_var))
    plt.scatter(range(len(importance_var)), importance_var)
    plt.title("clustering of importance variances per arm")
    plt.savefig("imp_variances.png")

    # Question 3: Does best-arm identification even make sense?
    mu = torch.tensor(A @ x)
    softmax_vals = F.softmax(mu, dim=0)
    sorted_vals = np.sort(softmax_vals)[::-1]
    cumsum = np.cumsum(sorted_vals)

    print("total entries: ", len(mu))
    print("minimum entry that's greater than : ", np.argmax(cumsum > 0.5))
    plt.scatter(range(len(mu)), cumsum)
    plt.title("cumsum of softmax values")
    plt.savefig("softmax values.png")

def get_adaptive_forward(
        model
) -> Callable:
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
        use_importance: bool = False,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        TODO(@Ryank): Add documentation

        :param input_ids:
        :param past_key_values:
        :param attention_mask:
        :param token_type_ids:
        :param position_ids:
        :param head_mask:
        :param inputs_embeds:
        :param encoder_hidden_states:
        :param encoder_attention_mask:
        :param labels:
        :param use_cache:
        :param output_attentions:
        :param output_hidden_states:
        :param return_dict:
        :param use_importance:
        :param verbose:
        :return:
        """
        labels = None  # we're only concerned with the log likelihood
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
        flattened_states = hidden_states.view(
            -1, hidden_states.size(-1)
        )  # TODO: currently only supports batch = 1

        A = model.lm_head.weight.data.cpu().numpy()  # size = (embed, vocab_size)
        x = flattened_states[-1].cpu().numpy()

        if debug:
            debug()

        # [NOTE] this is where our algorithm is being called!
        best_arms, z, adaptive_budget = ada_softmax(
            A=A,
            x=x,
            # samples_for_sigma=flattened_states.shape[0],
            samples_for_sigma=None,  # this finds the exact sigma <- debugging purposes
            beta=WIKITEXT_BETA,  # mu is very spiky
            verbose=verbose,
        )
        likelihood = torch.tensor(z)
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
    adaptive_model = copy.deepcopy(naive_model)
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
        target_id = tokens[:, -1].cpu()  # TODO: currently only allows cpu support

        with torch.no_grad():
            naive_logits = naive_model(input_ids, labels=None, return_dict=False)[0]
            flattened_naive_logits = naive_logits.view(-1, naive_logits.size(-1))
            naive_ll = F.softmax(
                WIKITEXT_BETA * flattened_naive_logits, dim=1
            )  # TODO: should be (batch_size, 1)
            naive_budget = naive_shape[0] * naive_shape[1]

            naive_ll = naive_ll[-1, target_id]  # just for the target
            adaptive_ll, adaptive_budget = adaptive_model(
                input_ids, labels=None, verbose=verbose
            )

        # CELoss averages the losses. But, we're only comparing likelihood
        naive_lls.append(naive_ll)
        adaptive_lls.append(adaptive_ll[target_id])

        if end_loc == seq_len:
            break
        print("\n")

    is_correct, gain = None, None
    if exp_type == "both" or exp_type == "correctness":
        is_correct = check_correctness(naive_lls, adaptive_lls, verbose=True)
    if exp_type == "both" or exp_type == "gains":
        true_mu = flattened_naive_logits
        gain = get_gains(
            true_mu.cpu().numpy(), naive_budget, adaptive_budget, verbose=True
        )

    print(f"==> Experiment {exp_type} is {is_correct} with gain {gain}")


if __name__ == "__main__":
    run_experiment(
        num_samples=10,
        model_id="gpt2-xl",
        verbose=True,
    )
