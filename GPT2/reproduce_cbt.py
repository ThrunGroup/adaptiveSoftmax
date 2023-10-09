from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm
import torch
import ipdb

softmax = torch.nn.functional.softmax

# Load pre-trained model and tokenizer
model_name = "gpt2"  # options: "gpt2", "gpt2-medium", "gpt2-large", or "gpt2-xl"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model.eval()  
model.to(device)
dataset = load_dataset("cbt", "CN")


def predict_missing_word(context, question, options, answer):
    pre_mask_text, post_mask_text = question.split("XXXXX")
    context_text = " ".join([sent.strip() for sent in context])
    input_prefix = context_text + " " + pre_mask_text.strip()

    best_option = None
    best_score = float('-inf')

    # Compute the logits for the input_prefix
    prefix_ids = tokenizer.encode(input_prefix, return_tensors='pt').to(device)
    with torch.no_grad():
        output = model(prefix_ids)  # get past_key_values for the next predictions
        prefix_logits, past = output['logits'], output['past_key_values']

    for option in options:
        option_id = tokenizer.encode(option, add_prefix_space=True)[0]
        likelihood = prefix_logits[0, -1, option_id]  # Log likelihood of the option given the input_prefix
        
        # Start the running sequence with the option
        running_ids = tokenizer.encode(option, return_tensors='pt').to(device)
        past_temp = past  # temporary past

        for word in post_mask_text.split():
            word_id = tokenizer.encode(word, add_prefix_space=True, return_tensors='pt').to(device)
            # Concatenate with the running sequence
            running_ids = torch.cat([running_ids, word_id], dim=1)
            with torch.no_grad():
                output = model(running_ids, past_key_values=past_temp)
                logits, past_temp = output['logits'], output['past_key_values']
            likelihood += logits[0, -1, word_id[0, 0]]

        # Check if this combined likelihood is higher than the best seen so far
        if likelihood > best_score:
            best_score = likelihood
            best_option = option

    print("best option is: ", best_option)
    return best_option == answer


def main():
    test_set = dataset['validation']
    sentences_list = test_set['sentences']
    question_list = test_set['question']
    options_list = test_set['options']
    answer_list = test_set['answer']

    correct_predictions = 0
    #total_samples = len(test_set) 
    total_samples = 5

    for i in tqdm(range(total_samples), desc="Predicting", ncols=100):     
        if predict_missing_word(sentences_list[i], question_list[i], options_list[i], answer_list[i]):
            correct_predictions += 1

    accuracy = correct_predictions / total_samples
    print(f"\nAccuracy on {total_samples} samples from the test set: {accuracy:.2f}")

if __name__ == "__main__":
    main()
