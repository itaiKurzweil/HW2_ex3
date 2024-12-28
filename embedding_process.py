import openai
import pickle
import pandas as pd
import os

# Set your OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found in environment variables.")
openai.api_key = api_key

# Load the CSV file
file_path = r"C:\Users\itaik\OneDrive\שולחן העבודה\Year c\AI dev\HW2\EX3\imdb_tvshows - imdb_tvshows.csv"
tv_shows_data = pd.read_csv(file_path)

# Create a dictionary to store embeddings
embeddings = {}

def generate_embeddings(description):
    """Generate embeddings using the new OpenAI API."""
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=description
    )
    return response["data"][0]["embedding"]


# Generate embeddings for each show
for _, row in tv_shows_data.iterrows():
    title = row["Title"]
    description = row["Description"]
    print(f"Generating embedding for: {title}")
    embeddings[title] = generate_embeddings(description)

# Save embeddings to a pickle file
output_file = "tv_show_embeddings.pkl"
with open(output_file, "wb") as file:
    pickle.dump(embeddings, file)

print(f"Embeddings saved to {output_file}")
