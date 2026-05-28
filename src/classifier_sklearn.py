"""Part 1: classical TF-IDF + LogisticRegression text classifier.

Fill in the TODO blocks below. The goal is to build a small,
self-contained scikit-learn pipeline that learns to map Polish text
commands to one of the five robot actions.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class SklearnClassifier:
    def __init__(self):
        # TODO 1: Build a sklearn Pipeline with two steps:
        #   - a TfidfVectorizer (good defaults: lowercase=True, ngram_range=(1, 2))
        #   - a LogisticRegression (good defaults: max_iter=1000)
        # Hint:
        #   self.pipeline = Pipeline([
        #       ("vec", TfidfVectorizer(...)),
        #       ("clf", LogisticRegression(...)),
        #   ])
        self.pipeline = None  # <-- replace

    def train(self, csv_path):
        """Train the pipeline on the provided CSV (columns: text, action)."""
        # TODO 2: Load the CSV with pandas (use the `csv_path` argument).
        df = None  # <-- replace

        # TODO 3: Split the data into train and test sets.
        # Example: train_test_split(df["text"], df["action"], test_size=0.2, random_state=42)
        x_train, x_test, y_train, y_test = None, None, None, None  # <-- replace

        # TODO 4: Fit the pipeline on the training data.

        # TODO 5: Predict on the test set, compute accuracy with accuracy_score,
        #         and print it so you can see how well the model fits.

    def predict(self, text):
        """Return (action, confidence) for a single input text.

        Confidence here is the maximum predicted class probability.
        """
        # TODO 6: Get the predicted label using self.pipeline.predict([text]).
        # TODO 7: Get the class probabilities using self.pipeline.predict_proba([text])
        #         and take the maximum as the confidence.
        action = "stop"     # <-- replace
        confidence = 0.0    # <-- replace
        return action, float(confidence)
