import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

# Load pre-trained model and tokenizer
model_name = "gpt2"  # options: "gpt2", "gpt2-medium", "gpt2-large", or "gpt2-xl"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

model.eval()  
model.to('cpu')
dataset = load_dataset("cbt", "CN") 
# the original paper has plots for "CN" and "NE"

def predict_missing_word(context, question, options, verbose=True):
    """Predicts the missing word in the question using the model."""
    # import ipdb; ipdb.set_trace()
    input_text = context + " " + question     # this is how they do it in original ICLR 2016 paper
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    
    with torch.no_grad():
        output = model.generate(input_ids, max_length=len(input_ids[0]) + 1, pad_token_id=tokenizer.eos_token_id, do_sample=False)
    
    predicted_token = output[0][-1].item()
    predicted_word = tokenizer.decode([predicted_token], skip_special_tokens=True)
    
    if verbose:
        print("predicted word: ", predicted_word)
        print("options are: ", options)

    return predicted_word in options


def main():
    """
    From GPT paper: 
        "CBT reports accuracy on an automatically constructed
        cloze test where the task is to predict which of 10 possible
        choices for an omitted word is correct"
    """
    sentences_list = dataset['test']['sentences']
    question_list = dataset['test']['question']
    options_list = dataset['test']['options']

    correct_predictions = 0
    #total_samples = len(dataset['test']) 
    total_samples = 5

    for i in range(total_samples):      
        if predict_missing_word(sentences_list[i][0], question_list[i], options_list[i]):
            correct_predictions += 1

    accuracy = correct_predictions / total_samples
    print(f"Accuracy on {total_samples} samples from the test set: {accuracy:.2f}")

if __name__ == "__main__":
    main()