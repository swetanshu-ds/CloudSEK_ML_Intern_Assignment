## Importing the dependencies
print("Importing the dependencies")



import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify

from sklearn.feature_extraction.text import TfidfVectorizer

import os
import logging
import datetime


print()
print("Importing of the dependencies done")


# Create a folder for logs if it doesn't exist
log_folder = 'logs_headline'
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

    
print("Logs Folder created")    
    
# Configure the logging
log_file = os.path.join(log_folder, f'accuracy_log_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.log')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')







# Reading the dataset
print("Starting to read")
df = pd.read_csv(r"D:\CloudSEK\train.csv")
print()
print("Reading done")
print()
print("Data Preprocessing")
print()

logging.info("Reading the dataset done")
print("Drop the duplicates")
df = df.drop_duplicates()
df.dropna(inplace = True)
df.reset_index(inplace = True)
df.drop(columns =  'index',inplace = True)
df.dropna(inplace = True)

print("Making a tokenizer  object")   

# Tokenize and preprocess the dummy data
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

tokenizer.pad_token = tokenizer.eos_token


logging.info("Making a tokenizer  object is done")


# Define the device for the data tensors
print("Move the model to the appropriate device (CPU or GPU)")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.info("Moved the model to the appropriate device (CPU or GPU)")
# Tokenize and preprocess the dummy data on the GPU

print("Load pre-trained GPT-2 model and tokenizer")
print()
# Load pre-trained GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

print("Load pre-trained GPT-2 model and tokenizer is done ")
class HeadlineGenerator(nn.Module):
    def __init__(self, base_model):
        super(HeadlineGenerator, self).__init__()
        self.base_model = base_model.to(device)
        self.lm_head = nn.Linear(base_model.config.hidden_size, base_model.config.vocab_size, bias=True)

    def forward(self, input_ids, attention_mask, labels=None):
        # Forward pass for headline generation
        outputs = self.base_model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            labels=labels.to(device)  # Pass labels here
        )
        return outputs
    
    

print("Instantiate the custom model") 
print("Loading the pretrained model")
loaded_model = HeadlineGenerator(model)
model_save_path = "D:\CloudSEK\headline_generator_model.pth"
# Load the saved model checkpoint
checkpoint = torch.load(model_save_path,map_location=torch.device('cpu'))

# Load the state dictionaries into the new model
loaded_model.load_state_dict(checkpoint['model_state_dict'])

# Set the model to evaluation mode (important for inference)
loaded_model.eval()

print(f"Model loaded from {model_save_path}")

print()
print(" Set the device to 'cuda' if a GPU is available, else use 'cpu'")
# Set the device to 'cuda' if a GPU is available, else use 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model.to(device)

logging.info("Loading the model is done")


print("Building the flask application")
print()


app = Flask(__name__)


def generate_headline(loaded_model, tokenizer, input_text, max_length=50):
    # Manually pad the input text with spaces on the left
    while len(input_text) < max_length:
        input_text = " " + input_text

    # Encode the input text
    input_encoded = tokenizer.encode(input_text, truncation=True, max_length=max_length, return_tensors='pt').to(device)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # Create an attention mask to ignore padding tokens
    attention_mask = input_encoded != tokenizer.pad_token_id

    # Generate a headline using the original GPT-2 model (base_model)
    with torch.no_grad():
        output = loaded_model.base_model.generate(input_encoded, attention_mask=attention_mask, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95)

    # Decode the generated headline and return it as a string
    generated_headline = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_headline
    
print("Building the cosine similarity calculation function")

def calculate_cosine_similarity(reference, candidate):
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the vectorizer on the reference and candidate
    tfidf_matrix = vectorizer.fit_transform([reference, candidate])

    # Calculate the cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])

    return cosine_sim[0][0]

# Assuming you have a test dataset with summaries and reference headlines

# Initialize variables to keep track of evaluation results
cosine_similarity_scores = []
df = pd.read_csv(r'D:\CloudSEK\test.csv')
for i in range(len(df)):
    input_summary = df.iloc[i]["Summary"]
    reference_headline = df.iloc[i]["Headline"]

    # Generate a headline
    generated_headline = generate_headline(loaded_model, tokenizer, input_summary)

    # Calculate BLEU score for this sample
    cosine_similarity_score = calculate_cosine_similarity(reference_headline, generated_headline)

    cosine_similarity_scores.append(cosine_similarity_score)

avg_cosine_score = sum(cosine_similarity_scores)/len(cosine_similarity_scores)*100



logging.info(f'Headline generation Test Cosine Score: {avg_cosine_score}')    



# Define the route for prediction
@app.route("/predict", methods=["POST"])
def predict():  
    if request.method == "POST":
        data = request.json
        text = data["text"]
        Generate_Headline = generate_headline(loaded_model, tokenizer, text, max_length=50)

        return jsonify({"Generate_Headline Label": Generate_Headline})


# Run the app if this file is executed
if __name__ == "__main__":
    app.run(debug=True,port = 5000)