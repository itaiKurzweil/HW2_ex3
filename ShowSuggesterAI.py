from annoy import AnnoyIndex
import pickle
import numpy as np
import requests
from PIL import Image
import os
import time
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
        indices, distances = self.index.get_nns_by_vector(avg_vector, top_k, include_distances=True)
        recommendations = [
            (self.show_id_map[idx], round((1 - (dist / 2)) * 100, 2))  # Adjust percentage calculation
            for idx, dist in zip(indices, distances)
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

    def generate_lightx_image(self, prompt, output_filename):
        """
        Generate an image using the LightX API based on the given prompt.

        Parameters:
            prompt (str): The text prompt describing the image to generate.
            output_filename (str): The filename to save the generated image.

        Returns:
            str: The local path to the saved image or None if the generation failed.
        """
        base_url = "https://api.lightxeditor.com/external/api/v1/text2image"
        status_url = "https://api.lightxeditor.com/external/api/v1/order-status"
        api_key = os.getenv("Lightxapikey")

        if not api_key:
            raise ValueError("LightX API key not set. Please define Lightxapikey as an environment variable.")

        try:
            # Initiate image generation
            payload = {"textPrompt": prompt}
            response = requests.post(
                base_url,
                json=payload,
                headers={
                    "x-api-key": api_key,
                    "Content-Type": "application/json"
                }
            )
            response.raise_for_status()
            response_data = response.json()

            order_id = response_data.get("body", {}).get("orderId")
            if not order_id:
                raise ValueError("Failed to retrieve order ID from LightX API response.")

            # Poll for order status
            for attempt in range(5):
                status_response = requests.post(
                    status_url,
                    json={"orderId": order_id},
                    headers={
                        "x-api-key": api_key,
                        "Content-Type": "application/json"
                    }
                )
                status_response.raise_for_status()
                status_data = status_response.json()

                output_url = status_data.get("body", {}).get("output")

                if output_url:
                    # Download and save the image locally
                    image_data = requests.get(output_url).content
                    with open(output_filename, "wb") as file:
                        file.write(image_data)
                    print(f"Image saved as: {output_filename}")
                    return output_filename

                time.sleep(5)  # Wait before retrying

            print(f"Image generation failed for prompt: {prompt}")
            return None

        except Exception as e:
            print(f"An error occurred while generating the image: {e}")
            return None
