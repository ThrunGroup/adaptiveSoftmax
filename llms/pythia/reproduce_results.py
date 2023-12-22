import torch

from tqdm import tqdm
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from datasets import load_dataset
from pythia_constants import PYTHIA_70M

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device is ", device)

# setting up model and tokenizer
model = GPTNeoXForCausalLM.from_pretrained(
    PYTHIA_70M,
    revision="step3000",
    cache_dir="./pythia-70m-deduped/step3000",
).to(device)
max_length = model.config.max_length    # length of the context window

tokenizer = AutoTokenizer.from_pretrained(
  PYTHIA_70M,
  revision="step3000",
  cache_dir="./pythia-70m-deduped/step3000",
)

# retrieving dataset
test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
seq_len = encodings.input_ids.size(1)   # length of the sequence we're analyzing

# perplexity experiment --> should be 11.39
nlls = []
stride = 512
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        import ipdb; ipdb.set_trace()
        outputs = model(input_ids, labels=target_ids)


        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())
print("perplexity is ", ppl)