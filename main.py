from ShowSuggesterAI import ShowSuggesterAI

if __name__ == "__main__":
    suggester = ShowSuggesterAI()

    # Load embeddings
    try:
        suggester.load_embeddings("tv_show_embeddings.pkl")
    except FileNotFoundError:
        print("Embeddings file not found. Please ensure the file is available.")
        exit(1)

    while True:
        # Get user input
        user_input = input("Which TV shows did you really like watching? Separate them by a comma. Make sure to enter more than 1 show ")
        user_shows = [show.strip() for show in user_input.split(",")]

        # Match shows
        matches = suggester.match_shows(user_shows)

        # Confirm matches
        print("Making sure, do you mean:")
        for input_show, matched_show in matches.items():
            print(f"{input_show} -> {matched_show if matched_show else 'No match found'}")

        confirm = input("Is this correct? (y/n): ").strip().lower()
        if confirm == 'y':
            # Filter out unmatched shows
            liked_shows = [matched_show for matched_show in matches.values() if matched_show]
            if not liked_shows:
                print("Sorry about that. Lets try again, please make sure to write the names of the tv shows correctly")
                continue

            # Generate recommendations
            print("Great! Generating recommendations now...")
            recommendations = suggester.recommend_shows(liked_shows)

            print("Here are the TV shows that I think you would love:")
            for show, similarity in recommendations:
                print(f"{show} ({similarity}%)")

            # Generate two custom shows with distinct names and descriptions
            custom_shows = {
                "show1": {
                    "name": f"Epic Adventures Beyond {liked_shows[0]}",
                    "description": "A new frontier of excitement and heroics inspired by the essence of your favorite shows."
                },
                "show2": {
                    "name": f"Mysteries and Legends", 
                    "description": "A thrilling exploration of untold stories and enigmatic characters that redefine intrigue."
                }
            }

            # Display the custom shows
            print("\nI have also created just for you two shows which I think you would love.")
            print(f"Show #1 is based on the fact that you loved the input shows that you gave me.")
            print(f"Its name is {custom_shows['show1']['name']} and it is about {custom_shows['show1']['description']}.")
            print(f"Show #2 is based on the shows that I recommended for you.")
            print(f"Its name is {custom_shows['show2']['name']} and it is about {custom_shows['show2']['description']}.")
            print("\nHere are also the 2 TV show ads. Hope you like them!")

            break
        else:
            print("Sorry about that. Let's try again.")