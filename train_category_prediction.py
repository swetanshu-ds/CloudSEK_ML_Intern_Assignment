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
logging.info("logging basicConfig Created")

# Reading the dataset

print("Starting to read")

df = pd.read_csv(r"D:\CloudSEK\train.csv")

print()

print("Reading done")

logging.info("Reading the dataset done")

print("Drop the duplicates")

df = df.drop_duplicates()
df.dropna(inplace = True)
unique_categories = list(df['Category'].unique())

l =[]

print("Taking the 25% of each category to make the balanced dataset")

for i in (unique_categories):
    a = df[df['Category'] == i]
    a = a.sample(frac=0.25, random_state=42)
    l.append(a)
print()
print("Concat the dataframes")

df = pd.concat(l)
df.reset_index(inplace = True)
df.drop(columns =  'index',inplace = True)
df.dropna(inplace = True)

print("Data Preprocessing done")

logging.info("Data Preprocessing done")

#label_encoder = preprocessing.LabelEncoder() is used to transform categorical labels into numerical values, making them usable in machine learning models.

print("Starting the label encoding")

label_encoder =  preprocessing.LabelEncoder()
df["Category"] = df['Category']
df['Category_encoded'] = label_encoder.fit_transform(df['Category']).astype(int)

print("Ending the label encoding")

# Storing labelEncoded Values in a y variable

y = df['Category'].values

map_dic = dict(zip(df["Category_encoded"],df["Category"]))

print("Making a tokenizer  object and a BERT Model Object")   

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=len(label_encoder.classes_))

logging.info("Making a tokenizer  object and a BERT Model Object is done")

print()

train_texts, train_labels = df['Summary'].tolist(), df['Category_encoded'].tolist()

print("Tokenize the texts and convert them to input tensors")
print()

# Tokenize the texts and convert them to input tensors
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256, return_tensors='pt')
train_dataset = TensorDataset(train_encodings.input_ids, train_encodings.attention_mask, torch.tensor(train_labels))

logging.info("Tokenize the texts and convert them to input tensors")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


print("Move the model to the appropriate device (CPU or GPU)")
# Move the model to the appropriate device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


logging.info("Moved the model to the appropriate device (CPU or GPU)")


# Define the optimizer and criterion
print("Define the optimizer and criterion")
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

logging.info("Defined the optimizer and criterion")


print()

print("Starting the training loop - fine tuning")
print()
# Training Loop
def training_fine_tuning():
    num_epochs = 500
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{num_epochs} - Average Training Loss: {avg_train_loss:.4f}')


    
logging.info("Ending the training loop - fine tuning")
print()
print("Evaluation started")
# Evaluation
def evaluation():
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            _, predicted = torch.max(logits, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

print("Evaluation done")
logging.info("Evaluation done")            
            
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
logging.info(f'Category prediction Train Accuracy Score: {accuracy}')
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