"""Part 2: sentence-transformer similarity classifier.

This classifier does not need a labelled training set. It relies on
a multilingual pretrained model and compares the input embedding to
a small set of reference templates for each action.
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# A handful of reference phrases per action. These are NOT training
# data - the pretrained model already understands them. They simply
# serve as anchor points the user input is compared against.
TEMPLATES = {
    "forward":    ["do przodu", "naprzód", "jedź prosto"],
    "backward":   ["do tyłu", "cofnij", "wstecz"],
    "turn_left":  ["w lewo", "skręć w lewo", "obrót w lewo"],
    "turn_right": ["w prawo", "skręć w prawo", "obrót w prawo"],
    "stop":       ["stop", "zatrzymaj się", "stój"],
}


class SentenceTransformerClassifier:
    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        # TODO 1: Load the pretrained model:
        #   self.model = SentenceTransformer(model_name)
        self.model = None  # <-- replace

        # TODO 2: For each action, encode all of its template phrases with
        # self.model.encode(...). Store the resulting arrays in
        # self.template_embeddings so we do not recompute them on every call.
        # Expected layout: {"forward": <ndarray (n_templates, dim)>, ...}
        self.template_embeddings = {}

    def predict(self, text):
        """Return (action, similarity) for a single input text.

        Similarity is the maximum cosine similarity between the input and
        any of the chosen action's templates.
        """
        # TODO 3: Encode the input text:
        #   user_vec = self.model.encode([text])

        # TODO 4: For each action, compute cosine_similarity between user_vec
        # and that action's template embeddings. The score for the action is
        # the maximum value across its templates.

        # TODO 5: Return the action with the highest score and that score.
        action = "stop"   # <-- replace
        score = 0.0       # <-- replace
        return action, float(score)
