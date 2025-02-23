
import argparse
from models.gpt3 import generate_description

def main():
    parser = argparse.ArgumentParser(description="Generate detailed caption using GPT-3.")
    parser.add_argument("--prompt", type=str, required=True, help="The input prompt text.")
    args = parser.parse_args()
    
    caption = generate_description(args.prompt)
    print("Generated Caption:")
    print(caption)

if __name__ == "__main__":
    main()
