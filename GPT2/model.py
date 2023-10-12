import torch
import torch.nn as nn
from typing import Union, Tuple

from adaptive_softmax.adasoftmax import ada_softmax

device = "cpu"
model_id = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
max_length = model.config.n_positions
stride = 512
seq_len = encodings.input_ids.size(1)

nlls = []
prev_end_loc = 0

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

def get_accuracy_gains():

