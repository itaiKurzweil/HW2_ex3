import pytest
from ShowSuggesterAI import ShowSuggesterAI

suggester = ShowSuggesterAI()

def test_initialization():
    assert isinstance(suggester, ShowSuggesterAI)
    assert suggester.tv_show_embeddings == {}

def test_load_embeddings():
    # Mocking a method call
    filepath = "test_embeddings.pkl"
    try:
        suggester.load_embeddings(filepath)
    except Exception as e:
        pytest.fail(f"load_embeddings raised an exception: {e}")

def test_save_embeddings():
    filepath = "test_save.pkl"
    try:
        suggester.save_embeddings(filepath)
    except Exception as e:
        pytest.fail(f"save_embeddings raised an exception: {e}")

def test_recommend_shows():
    liked_shows = ["Breaking Bad", "Sherlock"]
    try:
        recommendations = suggester.recommend_shows(liked_shows)
        assert isinstance(recommendations, list)
    except Exception as e:
        pytest.fail(f"recommend_shows raised an exception: {e}")