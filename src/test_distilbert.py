from pathlib import Path
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "distilbert_sentiment_model"
)

print("Loading model from:")
print(MODEL_DIR)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)

model.eval()

print("\nModel loaded successfully!")
print("Model type:", model.__class__.__name__)
print("Number of labels:", model.config.num_labels)
print("Labels:", model.config.id2label)


# Test prediction
review = """
The movie was absolutely fantastic.
The acting was excellent and the story was engaging.
"""

inputs = tokenizer(
    review,
    return_tensors="pt",
    truncation=True,
    max_length=256
)

with torch.no_grad():
    outputs = model(**inputs)

probabilities = torch.softmax(
    outputs.logits,
    dim=1
)

predicted_class = torch.argmax(
    probabilities,
    dim=1
).item()

confidence = probabilities[0, predicted_class].item()

label_map = {
    0: "Negative",
    1: "Positive"
}

print("\nTest Review:")
print(review.strip())

print("\nPrediction:", label_map[predicted_class])
print("Confidence:", f"{confidence * 100:.2f}%")