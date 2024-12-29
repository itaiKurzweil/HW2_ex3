from ShowSuggesterAI import ShowSuggesterAI
import openai
import os

# Function to generate custom shows using OpenAI

def ask_gpt_4_mini(prompt):
    """
    Send a prompt to OpenAI's GPT-4 model and return its response.

    Parameters:
        prompt (str): The prompt/question to send to GPT-4.

    Returns:
        str: The response from GPT-4.
    """
    try:
        openai.api_key = os.getenv("OPENAI_API_KEY")

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an assistant helping to create custom TV shows."},
                {"role": "user", "content": prompt}
            ]
        )

        # Extract the reply from the response object
        reply = response['choices'][0]['message']['content'].strip()
        return reply

    except openai.error.OpenAIError as e:
        return f"An error occurred: {str(e)}"


def generate_custom_shows(liked_shows, recommended_shows):
    """Generate two custom shows using OpenAI GPT."""
    # Construct prompts for the custom shows
    prompt = (
        f"Create two unique TV shows. The first is based on these liked shows: {', '.join(liked_shows)}. "
        f"The second is based on these recommended shows: {', '.join(recommended_shows)}. "
        "For each, provide a name and a brief description."
    )

    # Call GPT-4 Mini for the response
    response = ask_gpt_4_mini(prompt)

    # Parse response into two shows
    shows = response.split("\n\n")
    if len(shows) >= 2:
        show1 = shows[0].split("\n", 1)
        show2 = shows[1].split("\n", 1)

        return {
            "show1": {
                "name": show1[0].strip() if len(show1) > 0 else "Unknown Show",
                "description": show1[1].strip() if len(show1) > 1 else "No description available."
            },
            "show2": {
                "name": show2[0].strip() if len(show2) > 0 else "Unknown Show",
                "description": show2[1].strip() if len(show2) > 1 else "No description available."
            }
        }
    else:
        return {
            "show1": {"name": "Error in GPT Response", "description": "Could not parse show 1."},
            "show2": {"name": "Error in GPT Response", "description": "Could not parse show 2."}
        }

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

            # Generate two custom shows using OpenAI API
            custom_shows = generate_custom_shows(liked_shows, [rec[0] for rec in recommendations])

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