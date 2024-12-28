from thefuzz import process

class ShowSuggesterAI:
    def __init__(self):
        # Sample list of popular TV shows
        self.popular_shows = ["Game of Thrones", "Lupin", "The Witcher", "Breaking Bad", "Sherlock", "Dark"]

    def get_user_input(self, user_input):
        """
        Processes user input provided as a string.
        """
        shows = [show.strip() for show in user_input.split(",") if show.strip()]
        if len(shows) < 2:
            return None, "Please enter at least two TV shows."
        return shows, None

    def validate_input(self, user_shows, confirm):
        """
        Validates user input and confirms corrections.
        """
        corrected_shows = []
        for show in user_shows:
            match, confidence = process.extractOne(show, self.popular_shows)
            corrected_shows.append((match, confidence))
        
        if confirm.lower() == "y":
            return [match for match, _ in corrected_shows], None
        else:
            return None, "Sorry about that. Let's try again."
