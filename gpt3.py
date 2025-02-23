import openai
import os
import yaml

def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def generate_description(prompt):
    config = load_config()
    gpt3_config = config.get("gpt3", {})
    
    response = openai.Completion.create(
        engine=gpt3_config.get("engine", "text-davinci-003"),
        prompt=f"Expand the following idea into a vivid, detailed description:\n\n{prompt}",
        max_tokens=gpt3_config.get("max_tokens", 100),
        temperature=gpt3_config.get("temperature", 0.8)
    )
    description = response.choices[0].text.strip()
    return description

if __name__ == "__main__":
    # Example usage
    prompt = "A futuristic cityscape at dusk"
    caption = generate_description(prompt)
    print("Generated Caption:", caption)
