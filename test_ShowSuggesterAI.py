import pytest
from ShowSuggesterAI import ShowSuggesterAI
import pickle
import numpy as np

suggester = ShowSuggesterAI()

def test_initialization():
    assert isinstance(suggester, ShowSuggesterAI), "Object is not an instance of ShowSuggesterAI."
    assert suggester.tv_show_embeddings == {}, "tv_show_embeddings should be initialized as an empty dictionary."

def test_load_embeddings(tmp_path):
    test_data = {"show1": [0.1, 0.2, 0.3]}
    test_file = tmp_path / "test_embeddings.pkl"
    with open(test_file, 'wb') as file:
        pickle.dump(test_data, file)

    suggester.load_embeddings(test_file)
    assert suggester.tv_show_embeddings == test_data, "Loaded embeddings do not match the expected data."

def test_load_embeddings_file_not_found():
    with pytest.raises(FileNotFoundError, match="File at non_existent_file.pkl not found."):
        suggester.load_embeddings("non_existent_file.pkl")

def test_save_embeddings(tmp_path):
    suggester.tv_show_embeddings = {"show2": [0.4, 0.5, 0.6]}
    test_file = tmp_path / "test_save.pkl"

    suggester.save_embeddings(test_file)
    with open(test_file, 'rb') as file:
        saved_data = pickle.load(file)

    assert saved_data == suggester.tv_show_embeddings, "Saved embeddings do not match the expected data."

def test_match_shows():
    suggester.tv_show_embeddings = {"Game of Thrones": [0.1, 0.2, 0.3], "Breaking Bad": [0.4, 0.5, 0.6]}
    input_shows = ["game of thrones", "breaking bad"]
    matches = suggester.match_shows(input_shows)
    assert matches == {"game of thrones": "Game of Thrones", "breaking bad": "Breaking Bad"}, "Matching failed."

def test_recommend_shows():
    suggester.tv_show_embeddings = {
        "Breaking Bad": [0.1, 0.2, 0.3],
        "Sherlock": [0.4, 0.5, 0.6],
        "Dark": [0.7, 0.8, 0.9]
    }
    liked_shows = ["Breaking Bad"]

    recommendations = suggester.recommend_shows(liked_shows)
    assert isinstance(recommendations, list), "Recommendations should be a list."
    assert len(recommendations) == 2, "Recommendations list length is incorrect."
    assert all(isinstance(r[1], float) for r in recommendations), "Recommendations should have similarity scores."

def test_recommend_shows_empty_input():
    with pytest.raises(ValueError, match="Liked shows list cannot be empty."):
        suggester.recommend_shows([])