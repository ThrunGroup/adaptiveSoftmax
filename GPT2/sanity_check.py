from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm
import torch

softmax = torch.nn.functional.softmax

# Load pre-trained model and tokenizer
model_name = "gpt2"  # options: "gpt2", "gpt2-medium", "gpt2-large", or "gpt2-xl"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

model.eval()  
model.to('cuda')
dataset = load_dataset("cbt", "CN")

def predict_missing_word(context, question, options, answer):
    """
    Task description from original paper:
        Predict which of 10 possible choices for an omitted word is correct. 
        Do this by computing the probability of each choice and the rest of the sentence conditioned on this choice according to the LM, 
        and predict the one with the highest probability.
    """
    # prefix
    pre_mask_text, post_mask_text = question.split("XXXXX")
    context_text = " ".join([sent.strip() for sent in context])
    input_prefix = context_text + " " + pre_mask_text.strip()

    post_text_ids = tokenizer.encode(post_mask_text[0], return_tensors='pt')
    post_mask_words = post_mask_text.split()

    #import ipdb; ipdb.set_trace()

    # Calculate likelihood of the rest of the sentence for each option
    best_option = None
    best_score = float('-inf')

    for option in options:
        total_likelihood_logit = 0.0

        word_list = [option] + post_mask_words

        indices_list = tokenizer.encode(' '.join(word_list))


        input_ids = tokenizer.encode(input_prefix, return_tensors='pt').to('cuda')

        with torch.no_grad():
            logits = model(input_ids).logits

        prior_logit = torch.sum(logits[0], 0)[indices_list[0]]  #to take different sum per words into account, normalize them to the same denominator, but might be bad practice?
        total_likelihood_logit += prior_logit

        #import ipdb; ipdb.set_trace()

        #candidate_text = input_prefix + option + post_mask_text
        candidate_text = input_prefix
        for i, word in enumerate(word_list[:-1]):
            #print(i)

            candidate_text = candidate_text + ' ' + word

            input_ids = tokenizer.encode(input_prefix, return_tensors='pt').to('cuda')

            with torch.no_grad():
                logits = model(input_ids).logits

            likelihood_logit = torch.nn.functional.normalize(torch.sum(logits[0], 0), dim=0)[indices_list[i+1]] #Evaluate the likelihood of the word that originally comes next to the sentence
            likelihood_logit = torch.sum(logits[0], 0)[indices_list[i+1]]
            total_likelihood_logit += likelihood_logit

        """
        input_ids = tokenizer.encode(candidate_text, return_tensors='pt') #Why use pt here? What does it mean?

        with torch.no_grad():
            logits = model(input_ids).logits

        score = logits[0].sum().item()
        if score > best_score:
            best_score = score
            best_option = option
        """
        if total_likelihood_logit > best_score:
            best_score = total_likelihood_logit
            best_option = option

    return best_option == answer  # Assuming the answer is a single word immediately after "XXXXX"

def main():
    sentences_list = dataset['test']['sentences']
    question_list = dataset['test']['question']
    options_list = dataset['test']['options']
    answer_list = dataset['test']['answer']

    correct_predictions = 0
    total_samples = 50  # Change this to len(dataset['test']) for the full dataset

    for i in tqdm(range(total_samples), desc="Predicting", ncols=100):     
        if predict_missing_word(sentences_list[i], question_list[i], options_list[i], answer_list[i]):
            correct_predictions += 1

    accuracy = correct_predictions / total_samples
    print(f"\nAccuracy on {total_samples} samples from the test set: {accuracy:.2f}")

if __name__ == "__main__":
    main()
