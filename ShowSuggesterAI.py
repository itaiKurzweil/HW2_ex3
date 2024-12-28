import pickle
from thefuzz import process
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ShowSuggesterAI:
    def __init__(self):
        """Initialize the ShowSuggesterAI with necessary attributes."""
        self.tv_show_embeddings = {}

    def load_embeddings(self, filepath):
        """Load embeddings from a pickle file."""
        try:
            with open(filepath, 'rb') as file:
                self.tv_show_embeddings = pickle.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"File at {filepath} not found.")
        except Exception as e:
            raise Exception(f"An error occurred while loading embeddings: {e}")

    def save_embeddings(self, filepath):
        """Save embeddings to a pickle file."""
        try:
            with open(filepath, 'wb') as file:
                pickle.dump(self.tv_show_embeddings, file)
        except Exception as e:
            raise Exception(f"An error occurred while saving embeddings: {e}")

    def match_shows(self, input_shows):
        """Match user-inputted shows to the closest shows in the dataset."""
        matched_shows = {}
        for show in input_shows:
            best_match, score = process.extractOne(show, self.tv_show_embeddings.keys())
            matched_shows[show] = best_match if score > 80 else None
        return matched_shows

    def recommend_shows(self, liked_shows):
        """Generate recommendations based on liked shows."""
        if not liked_shows:
            raise ValueError("Liked shows list cannot be empty.")

        liked_vectors = [
            self.tv_show_embeddings[show] for show in liked_shows if show in self.tv_show_embeddings
        ]
        if not liked_vectors:
            raise ValueError("None of the liked shows were found in the dataset.")

        avg_vector = np.mean(liked_vectors, axis=0).reshape(1, -1)
        recommendations = []

        for show, embedding in self.tv_show_embeddings.items():
            if show not in liked_shows:
                similarity = cosine_similarity(avg_vector, np.array(embedding).reshape(1, -1))[0][0]
                recommendations.append((show, similarity))

        recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:5]
        return [(show, round(similarity * 100, 2)) for show, similarity in recommendations]