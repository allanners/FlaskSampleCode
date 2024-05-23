# Load required libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F  # Import torch.nn.functional
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import classification_report
from datasets import load_dataset
from flask import Flask, request, jsonify, render_template

# Defining CNN Architecture
"""
vocab_size: The size of the vocabulary (number of unique words).
embedding_dim: The dimensionality of the word embeddings.
num_filters: The number of filters (or feature maps) in the convolutional layers.
filter_sizes: A list of filter sizes for the convolutional layers.
output_dim: The dimensionality of the output (number of classes).
dropout: The dropout probability to prevent overfitting.
"""

# Get the class labels
class_labels = ['negative', 'positive']

# Define max length for padding
max_len = 128

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, output_dim, dropout):
        super(CNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim) # Embedding layer, convert input word indices into dense word vectors of dimension embedding_dim.

        # This line creates a list of convolutional layers (nn.Conv2d).
        # Each convolutional layer has in_channels=1 (since we are dealing with text data, which has one channel),
        # out_channels=num_filters (the number of output channels or feature maps),
        # and kernel_size=(fs, embedding_dim) (the size of the convolutional kernel).
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        # Creates a fully connected (linear) layer (nn.Linear) that takes the flattened output from the convolutional layers
        # maps it to the output dimension (output_dim), which represents the number of classes
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [batch size, sent len]

        # embedded = [batch size, sent len, emb dim]
        embedded = self.embedding(text) # passes input text to embedded layer

        # embedded = [batch size, 1, sent len, emb dim]
        embedded = embedded.unsqueeze(1)

        # convolutional layer
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # pooled = [batch size, num_filters]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # cat = [batch size, num_filters * len(filter_sizes)]
        cat = self.dropout(torch.cat(pooled, dim=1))

        # output = [batch size, output dim]
        output = self.fc(cat)

        return output

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Define hyperparameters
VOCAB_SIZE = tokenizer.vocab_size
EMBEDDING_DIM = 100
NUM_FILTERS = 100
FILTER_SIZES = [3, 4, 5]
OUTPUT_DIM = 2
DROPOUT = 0.5
WEIGHT_DECAY = 0.0001  # L2 regularization hyperparameter

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(VOCAB_SIZE, EMBEDDING_DIM, NUM_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT).to(device)

# Predict Inputs
checkpoint = torch.load('magtibaymodel\model_100_SST2_SPLIT.pth')
model.load_state_dict(checkpoint['model_state_dict'])
epoch = checkpoint['epoch']
train_loss = checkpoint['train_loss']
train_accuracy = checkpoint['train_accuracy']
val_loss = checkpoint['val_loss']
val_accuracy = checkpoint['val_accuracy']
accuracy = checkpoint['accuracy']
model.eval()

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)

    # Make prediction using the loaded model
    inputs = tokenizer(data['text'], truncation=True, max_length=max_len, padding='max_length', return_tensors='pt')
    inputs = inputs.to(device)
    with torch.no_grad():
        output = model(inputs['input_ids'])
    _, predicted = torch.max(output, 1)
    predicted_label = class_labels[predicted.item()]

    # Send back to the client
    output = {'prediction': predicted_label}
    return jsonify(output)

if __name__ == '__main__':
    app.run(port=5000, debug=True)

