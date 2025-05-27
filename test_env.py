from dotenv import load_dotenv
import os

print("Starting test...")
load_dotenv()
print("Environment loaded")
print("API Key:", os.getenv("OPENAI_API_KEY")) 