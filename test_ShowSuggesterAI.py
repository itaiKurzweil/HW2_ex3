import pytest
from ShowSuggesterAI import ShowSuggesterAI
import pickle
import os

suggester = ShowSuggesterAI()

def test_initialization():
    assert isinstance(suggester, ShowSuggesterAI), "Object is not an instance of ShowSuggesterAI."
    assert suggester.tv_show_embeddings == {}, "tv_show_embeddings should be initialized as an empty dictionary."

def test_load_embeddings(tmp_path):
    # Generate a dummy embedding of the correct dimension
    test_data = {"show1": [0.1] * 1536}  # 1536-dimensional vector
    test_file = tmp_path / "test_embeddings.pkl"
    with open(test_file, 'wb') as file:
        pickle.dump(test_data, file)

    suggester.load_embeddings(test_file)
    assert suggester.tv_show_embeddings == test_data, "Loaded embeddings do not match the expected data."


def test_load_embeddings_file_not_found():
    with pytest.raises(FileNotFoundError, match="File at non_existent_file.pkl not found."):
        suggester.load_embeddings("non_existent_file.pkl")

def test_recommend_shows():
    suggester.tv_show_embeddings = {
        "Breaking Bad": [0.1] * 1536,
        "Sherlock": [0.2] * 1536,
        "Dark": [0.3] * 1536,
        "Stranger Things": [0.25] * 1536,
        "Game of Thrones": [0.35] * 1536,
    }
    liked_shows = ["Breaking Bad"]

    recommendations = suggester.recommend_shows(liked_shows, top_k=3)  # Request 3 recommendations

    print("Recommendations:", recommendations)  # Debugging output

    assert isinstance(recommendations, list), "Recommendations should be a list."
    assert len(recommendations) >= 1, "Recommendations list should not be empty."
    assert all(isinstance(r[1], float) for r in recommendations), "Recommendations should have similarity scores."




def test_recommend_shows_empty_input():
    with pytest.raises(ValueError, match="Liked shows list cannot be empty."):
        suggester.recommend_shows([])

def test_match_shows():
    suggester.tv_show_embeddings = {"Game of Thrones": [0.1, 0.2, 0.3], "Breaking Bad": [0.4, 0.5, 0.6]}
    input_shows = ["game of thrones", "breaking bad"]
    matches = suggester.match_shows(input_shows)
    assert matches == {"game of thrones": "Game of Thrones", "breaking bad": "Breaking Bad"}, "Matching failed."

def test_generate_lightx_image_valid(tmp_path):
    suggester = ShowSuggesterAI()
    prompt = "Test prompt for a valid poster."
    output_filename = tmp_path / "test_image_valid.png"

    result = suggester.generate_lightx_image(prompt, str(output_filename))

    # Assert the result
    if result is not None:
        assert os.path.exists(result), "Image file was not created as expected."
        assert os.path.getsize(result) > 0, "Image file was created but is empty."
    else:
        pytest.skip("Skipping test: LightX API might not be accessible or valid for this environment.")

def test_generate_lightx_image_invalid(tmp_path):
    suggester = ShowSuggesterAI()
    prompt = ""  # Invalid prompt
    output_filename = tmp_path / "test_image_invalid.png"

    result = suggester.generate_lightx_image(prompt, str(output_filename))

    # Assert the result
    assert result is None, "Image generation should fail with an invalid prompt."
