from ShowSuggesterAI import ShowSuggesterAI

def test_get_user_input_valid():
    suggester = ShowSuggesterAI()
    result, message = suggester.get_user_input("Breaking Bad, Game of Thrones")
    assert result == ["Breaking Bad", "Game of Thrones"]
    assert message is None, "There should be no error message for valid input."

def test_validate_input_correct():
    suggester = ShowSuggesterAI()
    user_shows = ["gem of throns", "lupan", "witcher"]
    result, message = suggester.validate_input(user_shows, "y")
    assert result == ["Game of Thrones", "Lupin", "The Witcher"]
    assert message is None, "There should be no error message if user confirms corrections."
