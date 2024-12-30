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


def generate_custom_shows(shows):
    """Generate two custom shows using OpenAI GPT."""
    # Construct prompts for the custom shows
    prompt = (
        f"Generate a new TV show./n"
        f"The show should include 'Name' and 'Description'./n"
        f"The show should be based on the following TV shows: {shows}./n"
    )

    # Call GPT-4 Mini for the response
    response = ask_gpt_4_mini(prompt)

    # Parse the response to extract name and description
    try:
        showname, showdescription = response.split(',', 1)
        return showname.strip(), showdescription.strip()
    except ValueError:
        return "Error in response", "Response could not be parsed"

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
            print("I have also created just for you two shows which I think you would love.")
            for i in range(2):
                custom_show_name, custom_show_description = generate_custom_shows(liked_shows if i == 0 else [rec[0] for rec in recommendations])
                print(f"\nShow #{i + 1}:")
                if i==0:
                    print("is based on the fact that you loved the input shows that you gave me.")
                else:
                    print("is based on the shows that I recommended for you.")
                print(f"{custom_show_name}")
                print(f" {custom_show_description}")

            break
        else:
            print("Sorry about that. Let's try again.")