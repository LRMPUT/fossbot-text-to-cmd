"""Part 2 - solution: sentence-transformer similarity classifier.

This is the filled-in reference version of `src/classifier_st.py`.
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# A handful of reference phrases per action. These are NOT training
# data - the pretrained model already understands them. They simply
# serve as anchor points the user input is compared against.
TEMPLATES = {
    "forward":    ["go forward", "forward", "ahead"],
    "backward":   ["go back", "backwards", "reverse"],
    "turn_left":  ["turn left", "left", "rotate left"],
    "turn_right": ["turn right", "right", "rotate right"],
    "stop":       ["stop", "halt", "stay still"],
}


class SentenceTransformerClassifier:
    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)
        self.template_embeddings = {
            action: self.model.encode(phrases)
            for action, phrases in TEMPLATES.items()
        }

    def predict(self, text):
        """Return (action, similarity) for a single input text."""
        user_vec = self.model.encode([text])

        best_action, best_score = None, -1.0
        for action, embeddings in self.template_embeddings.items():
            score = cosine_similarity(user_vec, embeddings).max()
            if score > best_score:
                best_action, best_score = action, score

        return best_action, float(best_score)
