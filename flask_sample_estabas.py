# Load required libraries
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

model_path = "estabasmodel"
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Load the tokenizer (replace with the tokenizer you used during fine-tuning)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def interpret_label(predicted_label):
    if predicted_label == 0:
        return "Negative"
    elif predicted_label == 1:
        return "Positive"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)

    # Make prediction using the loaded model
    tokenized_text = tokenizer(data["text"], truncation=True, return_tensors="pt")
    outputs = model(**tokenized_text)
    predicted_label = outputs.logits.argmax(-1)
    interpretation = interpret_label(predicted_label.item())

    # Send back to the client
    output = {'prediction': interpretation}
    return jsonify(output)

if __name__ == '__main__':
    app.run(port=5000, debug=True)


