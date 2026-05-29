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
        # Pattern: self.pipeline = Pipeline([(name1, step1), (name2, step2)])
        # Docs:
        #   Pipeline           - https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
        #   TfidfVectorizer    - https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
        #   LogisticRegression - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        self.pipeline = None  # <-- replace

    def train(self, csv_path):
        """Train the pipeline on the provided CSV (columns: text, action)."""
        # TODO 2: Load the CSV with pandas (use the `csv_path` argument).
        # Docs: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
        df = None  # <-- replace

        # TODO 3: Split the data into train and test sets using train_test_split.
        # Use df["text"] as features and df["action"] as labels.
        # Use test_size=0.2 and random_state=42 for reproducible results.
        # Docs: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
        x_train, x_test, y_train, y_test = None, None, None, None  # <-- replace

        # TODO 4: Fit the pipeline on the training data.
        # Docs: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline.fit

        # TODO 5: Predict on the test set with self.pipeline.predict(...),
        #         compute accuracy with accuracy_score, and print it.
        # Docs:
        #   Pipeline.predict - https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline.predict
        #   accuracy_score   - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html

    def predict(self, text):
        """Return (action, confidence) for a single input text.

        Confidence here is the maximum predicted class probability.
        """
        # TODO 6: Get the predicted label using self.pipeline.predict(...).
        #         Note: the method expects a list of texts, not a single string.
        # TODO 7: Get the class probabilities using self.pipeline.predict_proba(...)
        #         and take the maximum value as the confidence.
        # Docs:
        #   Pipeline.predict       - https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline.predict
        #   Pipeline.predict_proba - https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline.predict_proba
        action = "stop"     # <-- replace
        confidence = 0.0    # <-- replace
        return action, float(confidence)
