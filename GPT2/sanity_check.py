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
num_added_tokens = tokenizer.add_special_tokens({"additional_special_tokens": [MASK_TOKEN]})
model.resize_token_embeddings(len(tokenizer))

model.eval()  
model.to('cpu')
dataset = load_dataset("cbt", "CN")

def predict_missing_word(context, question, options):
    mask_position = tokenizer.encode(MASK_TOKEN, add_special_tokens=False)[0]  # Get the token id for MASK_TOKEN
    context_ids = tokenizer.encode(" ".join(context), return_tensors='pt')
    question_ids = tokenizer.encode(question.replace("XXXXX", MASK_TOKEN), return_tensors='pt')
    input_ids = torch.cat([context_ids, question_ids], dim=1)

    if input_ids.shape[1] > model.config.n_positions:
        print("Input is too long for model")
        return False

    mask_index = torch.where(input_ids == mask_position)[1].item()  # Find the position of the MASK token
    with torch.no_grad():
        output = model.generate(input_ids, 1)  # Give some space for generation but you can adjust

    ipdb.set_trace()
    predicted_token = output[0][mask_index].item()
    predicted_word = tokenizer.decode([predicted_token], skip_special_tokens=True)
    print("this is prediction: ", predicted_word)
    print("these are options: ", options)
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
