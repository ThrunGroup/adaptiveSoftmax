from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
import torch
from tqdm import tqdm

# Load dataset and model from huggingface
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name).eval()
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Prepare input tokens
stride = 32
max_length = 1024
text = dataset["test"]["text"]
inputs = tokenizer(text, return_tensors="pt", max_length=max_length, stride=stride, truncation=True)

# Remove potential empty sequences
import ipdb; ipdb.set_trace()
non_empty_indices = [i for i, input_ids in enumerate(inputs["input_ids"]) if len(input_ids) > 0]
inputs["input_ids"] = inputs["input_ids"][non_empty_indices]
inputs["attention_mask"] = inputs["attention_mask"][non_empty_indices]

# Compute the perplexity over the tokenized test set
total_loss = 0.0
num_chunks = len(inputs["input_ids"])

with torch.no_grad():
    for i in tqdm(range(num_chunks)):
        input_ids = inputs["input_ids"][i].unsqueeze(0)
        attention_mask = inputs["attention_mask"][i].unsqueeze(0)
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        total_loss += outputs.loss.item()

avg_loss = total_loss / num_chunks
perplexity = torch.exp(torch.tensor(avg_loss)).item()

print(f"Perplexity: {perplexity}")
