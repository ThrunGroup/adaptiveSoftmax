from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm
import torch

import ipdb

# Load pre-trained model and tokenizer
MASK_TOKEN = "<MASK>"
model_name = "gpt2"  # options: "gpt2", "gpt2-medium", "gpt2-large", or "gpt2-xl"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
if tokenizer.add_special_tokens({"mask_token": MASK_TOKEN}) > 0:
    print("=> <MASK> token added...")      # making sure GPT2 recognizes the <MASK> token

model.resize_token_embeddings(len(tokenizer))
model.eval()  
model.to('cpu')
dataset = load_dataset("cbt", "CN")

def predict_missing_word(context, question, options):
    #ipdb.set_trace()
    mask_id = tokenizer.encode(MASK_TOKEN, add_special_tokens=False)[0]  # Get the token id for MASK_TOKEN
    context_ids = tokenizer.encode(" ".join(context), return_tensors='pt')
    question_ids = tokenizer.encode(question.replace("XXXXX", MASK_TOKEN), return_tensors='pt')
    input_ids = torch.cat([context_ids, question_ids], dim=1)
    mask_index = torch.where(input_ids == mask_id)[1].item()  

    with torch.no_grad():
        logits = model(input_ids, attention_mask=torch.ones_like(input_ids)).logits
        logits_at_mask = logits[0, mask_index]
        logits_at_mask[mask_id] = float('-inf')    # mask the MASK_TOKEN when predicting
        predicted_token = torch.argmax(logits_at_mask).item()

    predicted_word = tokenizer.decode([predicted_token], skip_special_tokens=True)
    print("question: ", question)
    print("predicted word: ", predicted_word)
    return predicted_word in options


def main():
    sentences_list = dataset['test']['sentences']
    question_list = dataset['test']['question']
    options_list = dataset['test']['options']

    correct_predictions = 0
    total_samples = 10
    #total_samples = len(dataset['test'])

    for i in tqdm(range(total_samples), desc="Predicting", ncols=100):     
        if predict_missing_word(sentences_list[i], question_list[i], options_list[i]):
            correct_predictions += 1

    accuracy = correct_predictions / total_samples
    print(f"\nAccuracy on {total_samples} samples from the test set: {accuracy:.2f}")

if __name__ == "__main__":
    main()
