"""
This Python script uses a transformer-based model (DistilBERT fine-tuned for sentiment analysis)
to perform sentiment classification. It includes two main sections:

1. Interactive Sentiment Prediction: Allows users to input a custom sentence and receive
   real-time sentiment classification (Positive or Negative).

2. Dataset-based Evaluation: Loads a synthetic dataset from a CSV file, predicts sentiment
   for each review using a transformer model, and evaluates the accuracy against ground truth
   labels using scikit-learn metrics.

The script uses Hugging Face's Transformers library for model inference and pandas for data handling.
"""

import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score

# Load sentiment pipeline (using DistilBERT fine-tuned on SST-2)
sentiment_pipeline = pipeline("sentiment-analysis")

# Function to return sentiment label name (Positive/Negative)
def get_sentiment_label_name(text):
    result = sentiment_pipeline(text)[0]['label']
    return result

# Function to return numeric sentiment label (1 for Positive, 0 for Negative)
def get_sentiment_numeric_label(text):
    label = sentiment_pipeline(text)[0]['label']
    if label == "POSITIVE":
        return 1
    else:
        return 0

# Interactive Sentiment Prediction
def interactive_sentiment():
    user_input = input("Enter text to analyze sentiment: ")
    result = get_sentiment_label_name(user_input)
    print("Sentiment:", result)

# Dataset-based Evaluation
def evaluate_on_dataset():
    # Load dataset and apply transformer-based sentiment classifier
    df = pd.read_csv('SyntheticDataset.csv')
    df['predicted_label'] = df['review_text'].apply(get_sentiment_numeric_label)

    # Evaluate accuracy
    accuracy = accuracy_score(df['label'], df['predicted_label'])
    print("\nDataset Evaluation")
    print("Accuracy:", accuracy)

    # Printing samples
    print(df[['review_text', 'label', 'predicted_label']])

if __name__ == "__main__":
    interactive_sentiment()
    evaluate_on_dataset()
