from ShowSuggesterAI import ShowSuggesterAI
import openai
import os
from PIL import Image
import time

# Function to generate custom shows using OpenAI
def ask_gpt_4_mini(prompt):
    try:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an assistant helping to create custom TV shows."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content'].strip()
    except openai.error.OpenAIError as e:
        return f"An error occurred: {str(e)}"

def generate_custom_show(liked_shows):
    prompt = (
        f"Generate a unique TV show idea.\n"
        f"Include the 'Name' and a 200-word 'Description'.\n"
        f"The show should be inspired by these: {', '.join(liked_shows)}."
    )
    response = ask_gpt_4_mini(prompt)
    try:
        name, description = response.split('\n', 1)
        name = name.replace("Name: ", "").strip('" ')
        description = description.replace("Description: ", "").strip('" ')
        return name, description
    except ValueError:
        raise ValueError(f"Failed to parse OpenAI response: {response}")

if __name__ == "__main__":
    suggester = ShowSuggesterAI()

    try:
        suggester.load_embeddings("tv_show_embeddings.pkl")
    except FileNotFoundError:
        print("Embeddings file not found. Please ensure the file is available.")
        exit(1)

    while True:
        user_input = input("Which TV shows did you really like watching? Separate them by a comma. Make sure to enter more than 1 show:")
        user_shows = [show.strip() for show in user_input.split(",")]

        matches = suggester.match_shows(user_shows)

        print("Making sure, do you mean:")
        for input_show, matched_show in matches.items():
            print(f"{input_show} -> {matched_show if matched_show else 'No match found'}")

        confirm = input("Is this correct? (y/n): ").strip().lower()
        if confirm == 'y':
            liked_shows = [matched_show for matched_show in matches.values() if matched_show]
            
            if not liked_shows:
                print("Sorry about that. Lets try again, please make sure to write the names of the tv shows correctly.")
                continue

            print("“Great! Generating recommendations now…")
            recommendations = suggester.recommend_shows(liked_shows)

            print("Here are the TV shows I think you would love:")
            for show, similarity in recommendations:
                print(f"{show} ({similarity}%)")

            print("""I have also created just for you two shows which I think you would love.
            Show #1 is based on the fact that you loved the input shows that you
            gave me.
            Show #2 is based on the shows that I recommended for you.
            Here are also the 2 tv show ads. Hope you like them!""")
            custom_shows = []
            for _ in range(2):  # Generate two custom shows
                try:
                    name, description = generate_custom_show(liked_shows)

                    # Clean and format the name and description
                    name = name.replace(':', '').replace('"', '').strip()
                    description = description.replace('"', '').replace("'", '').strip()
                    description = (description.split('.')[0][:400] + '...') if len(description) > 400 else description

                    custom_shows.append({"name": name, "description": description})
                    print(f"Name: {name} \n Description: {description}")
                    

                    # Prepare a simplified image prompt
                    image_prompt = f"Create a poster for the TV show '{name}' with the tagline: '{description}'."

                    image_filename = f"custom_show_{name.replace(' ', '_')}.png"
                    generated_image_path = suggester.generate_lightx_image(image_prompt, image_filename)
                    
                    if not generated_image_path:
                        print(f"Retrying with a simpler prompt for: {name}")
                        simplified_prompt = f"Create a poster for the TV show '{name}'."
                        generated_image_path = suggester.generate_lightx_image(simplified_prompt, image_filename)
                    
                    if generated_image_path:
                        img = Image.open(generated_image_path)
                        img.show()
                    else:
                        print(f"Failed to generate image for: {name}")
                except Exception as e:
                    print(f"Error generating custom show or image: {e}")
            break
        else:
            print("Sorry about that. Let's try again.")
