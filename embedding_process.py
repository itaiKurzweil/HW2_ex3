import pandas as pd
import openai
import pickle

# Set your OpenAI API key
openai.api_key = "your_openai_api_key"

def load_and_process_csv(file_path):
    """Load CSV and process titles and descriptions."""
    tv_shows_data = pd.read_csv(file_path)
    tv_shows = tv_shows_data[["Title", "Description"]]
    return tv_shows

def generate_embeddings(description):
    """Generate embeddings for a given description."""
    response = openai.Embedding.create(
        input=description,
        model="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]

def save_embeddings(tv_shows, output_file):
    """Generate and save embeddings for TV shows."""
    embeddings = {}
    for _, row in tv_shows.iterrows():
        title = row["Title"]
        description = row["Description"]
        embeddings[title] = generate_embeddings(description)
    with open(output_file, "wb") as file:
        pickle.dump(embeddings, file)
    print(f"Embeddings saved to {output_file}!")

if __name__ == "__main__":
    # Path to the CSV file
    file_path = r"C:\Users\itaik\OneDrive\שולחן העבודה\Year c\AI dev\HW2\EX3\imdb_tvshows - imdb_tvshows.csv"
    output_file = "tv_show_embeddings.pkl"

    # Process CSV and generate embeddings
    tv_shows = load_and_process_csv(file_path)
    save_embeddings(tv_shows, output_file)
