from annoy import AnnoyIndex
import pickle
import numpy as np
from thefuzz import process

class ShowSuggesterAI:
    def __init__(self, embedding_dim=1536):
        """Initialize with Annoy index and embedding dimension."""
        self.embedding_dim = embedding_dim
        self.tv_show_embeddings = {}
        self.index = AnnoyIndex(self.embedding_dim, 'angular')
        self.show_id_map = {}  # Maps Annoy index IDs to show names

    def load_embeddings(self, filepath):
        """Load embeddings from a pickle file and build the Annoy index."""
        try:
            with open(filepath, 'rb') as file:
                self.tv_show_embeddings = pickle.load(file)

            # Populate the Annoy index
            for i, (show, embedding) in enumerate(self.tv_show_embeddings.items()):
                self.index.add_item(i, embedding)
                self.show_id_map[i] = show

            self.index.build(10)  # Build the index with 10 trees
        except FileNotFoundError:
            raise FileNotFoundError(f"File at {filepath} not found.")
        except Exception as e:
            raise Exception(f"An error occurred while loading embeddings: {e}")

    def recommend_shows(self, liked_shows, top_k=7):
        """Generate recommendations based on liked shows using Annoy."""
        if not liked_shows:
            raise ValueError("Liked shows list cannot be empty.")

        liked_vectors = [
            self.tv_show_embeddings[show] for show in liked_shows if show in self.tv_show_embeddings
        ]
        if not liked_vectors:
            raise ValueError("None of the liked shows were found in the dataset.")

        # Calculate the average vector
        avg_vector = np.mean(liked_vectors, axis=0)

        # Find the nearest neighbors
        indices = self.index.get_nns_by_vector(avg_vector, top_k, include_distances=True)
        recommendations = [
            (self.show_id_map[idx], round((1 - dist) * 100, 2))
            for idx, dist in zip(*indices)
            if self.show_id_map[idx] not in liked_shows
        ]
        return recommendations

    def match_shows(self, input_shows):
        """Match user-inputted shows to the closest shows in the dataset."""
        matched_shows = {}
        for show in input_shows:
            best_match, score = process.extractOne(show, self.tv_show_embeddings.keys())
            matched_shows[show] = best_match if score > 80 else None
        return matched_shows
