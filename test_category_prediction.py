## Importing the dependencies
print("Importing the dependencies")

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from flask import Flask, request, jsonify
from sklearn.metrics import accuracy_score

import os
import logging
import datetime
print()
print("Importing of the dependencies done")

# Create a folder for logs if it doesn't exist
log_folder = 'logs_category'
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

print("Logs Folder creared")    
    
# Configure the logging
log_file = os.path.join(log_folder, f'accuracy_log_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.log')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Reading the dataset
print("Starting to read")
df = pd.read_csv(r"D:\CloudSEK\train.csv")
print()
print("Reading done")

logging.info("Reading the dataset done")

print("Drop the duplicates")

df = df.drop_duplicates()
df.dropna(inplace = True)

df.reset_index(inplace = True)
df.drop(columns =  'index',inplace = True)
df.dropna(inplace = True)




print("Starting the label encoding")
#label_encoder = preprocessing.LabelEncoder() is used to transform categorical labels into numerical values, making them usable in machine learning models.
label_encoder =  preprocessing.LabelEncoder()
df["Category"] = df['Category']
df['Category_encoded'] = label_encoder.fit_transform(df['Category']).astype(int)

print("Ending the label encoding")

print("Making a tokenizer  object and a BERT Model Object")   

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=len(label_encoder.classes_))

logging.info("Making a tokenizer  object and a BERT Model Object is done")
print()


print("Move the model to the appropriate device (CPU or GPU)")
# Move the model to the appropriate device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

logging.info("Moved the model to the appropriate device (CPU or GPU)")


print("Load the model")
print()
def load_model(model_class, num_labels, path):
    model = model_class.from_pretrained('bert-base-uncased', num_labels=num_labels)
    model.load_state_dict(torch.load(path))
    model.to(device)  # Move the model back to the appropriate device
    return model



# Load the pre-trained model
loaded_model = load_model(BertForSequenceClassification, len(label_encoder.classes_), 'D:\CloudSEK\pretrained_model_2.pt')

print("Loading the model is done")
logging.info("Loading the model is done")


print("Building the flask application")
print()

app = Flask(__name__)


print("Building the predict_single_input application")
print()

def predict_single_input(model, input_paragraph, device):
    model.eval()

    # Ensure the model and input tensors are on the same device
    model = model.to(device)

    # Tokenize the input paragraph
    inputs = tokenizer(input_paragraph, truncation=True, padding=True, max_length=256, return_tensors='pt')
    input_ids, attention_mask = inputs.input_ids.to(device), inputs.attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        _, predicted = torch.max(logits, 1)
        predicted_label = label_encoder.classes_[predicted.item()]

    return predicted_label


print("predict_single_input building is done")
logging.info("predict_single_input building is done")
print()

df = pd.read_csv(r'D:\CloudSEK\test.csv')

print("Building the accuracy checking system")
print("Accuracy building started")


y_pred  = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for i in range(len(df)):
    # Example usage of the predict_single_input function with the loaded model
    input_paragraph = df["Summary"].iloc[i]
    predicted_label = predict_single_input(loaded_model, input_paragraph,device)
    y_pred.append(predicted_label)
    
    
pred = df.iloc[:len(df)]['Category']
accuracy = accuracy_score(pred, y_pred)

print("Accuracy building completed")
# Log the accuracy score with a timestamp
logging.info(f'Test Accuracy Score Category Prediction : {accuracy}')
print("@app.route started")

@app.route("/predict", methods=["POST"])
def predict():  
    if request.method == "POST":
        data = request.json
        text = data["text"]
        Predict_Category = predict_single_input(model, text, device)

        return jsonify({"Predict_Category": Predict_Category})



# Run the app if this file is executed
if __name__ == "__main__":
    app.run(debug=True,port = 5000)