"""Part 1 - solution: classical TF-IDF + LogisticRegression text classifier.

This is the filled-in reference version of `src/classifier_sklearn.py`.
You can either copy this file over `src/classifier_sklearn.py` to run the
CLI as a sanity check, or compare with your own implementation after you
have finished writing it.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class SklearnClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ("vec", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000)),
        ])

    def train(self, csv_path):
        """Train the pipeline on the provided CSV (columns: text, action)."""
        df = pd.read_csv(csv_path)
        x_train, x_test, y_train, y_test = train_test_split(
            df["text"], df["action"], test_size=0.2, random_state=42
        )
        self.pipeline.fit(x_train, y_train)
        predictions = self.pipeline.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Test accuracy: {accuracy:.3f}")

    def predict(self, text):
        """Return (action, confidence) for a single input text.

        Confidence is the maximum predicted class probability.
        """
        action = self.pipeline.predict([text])[0]
        confidence = float(self.pipeline.predict_proba([text])[0].max())
        return action, confidence
