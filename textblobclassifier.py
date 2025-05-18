"""
This Python script performs sentiment analysis using TextBlob and is organized into two main parts: 
interactive sentiment prediction and dataset-based evaluation.

In the interactive section, users can input custom text to receive real-time sentiment feedback 
(positive, negative, or neutral), allowing for quick testing and experimentation.

In the dataset evaluation section, the script reads a synthetic CSV dataset, applies the sentiment 
analysis function to the review texts, and compares the predicted labels with actual labels to 
calculate accuracy using scikit-learn's metrics.

It prints the accuracy and a preview of the review texts alongside their original and predicted 
labels. The use of pandas ensures efficient data manipulation, and the combination of TextBlob 
and scikit-learn provides a simple yet powerful framework for evaluating sentiment models.
"""

import pandas as pd
from textblob import TextBlob
from sklearn.metrics import accuracy_score

# Function to return sentiment label name for a given text
def get_sentiment_label_name(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Function to return numeric sentiment label
def get_sentiment_numeric_label(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0 : 
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
    # Load dataset and get predicted labels
    df = pd.read_csv('SyntheticDataset.csv')
    df['predicted_label'] = df['review_text'].apply(get_sentiment_numeric_label)

    # Accuracy
    accuracy = accuracy_score(df['label'], df['predicted_label'])
    print("\nDataset Evaluation")
    print("Accuracy:", accuracy)

    # Printing samples
    print(df[['review_text', 'label', 'predicted_label']])

if __name__ == "__main__":
    interactive_sentiment()
    evaluate_on_dataset()
