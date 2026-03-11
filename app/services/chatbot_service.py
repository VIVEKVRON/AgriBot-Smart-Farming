import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. Load the hidden variables from the .env file
load_dotenv()

# 2. Fetch the specific API key
my_api_key = os.getenv("GEMINI_API_KEY")

if not my_api_key:
    print("Error: API key not found. Please check your .env file!")
else:
    # 3. Initialize the new genai Client
    client = genai.Client(api_key=my_api_key)
    print("API securely configured!")

# 4. Define the System Prompt (The Persona)
system_instruction = """
You are AgriBot, an intelligent and helpful agricultural assistant specifically designed to help Indian farmers. 
You must ONLY answer questions related to agriculture, farming, crop cultivation, fertilizers, plant diseases, irrigation, and Indian government agricultural schemes. 
If a user asks about anything else (like movies, programming, politics, or general trivia), politely decline and remind them that you are an agricultural assistant. 
Keep your answers clear, concise, and easy to understand for farmers.
"""

# 5. Configure the Chat Settings
config = types.GenerateContentConfig(
    system_instruction=system_instruction,
)

# 6. Start a chat session using the new syntax
chat = client.chats.create(
    model="gemini-2.5-flash",
    config=config
)

# 7. Create the interaction function
def ask_agribot(user_message):
    print(f"🧑‍🌾 Farmer: {user_message}")
    
    # Send the message to the LLM
    response = chat.send_message(user_message)
    
    print(f"🤖 AgriBot: {response.text}\n")
    print("-" * 50 + "\n")
    return response.text

# --- Let's Test It! ---
if __name__ == "__main__":
    ask_agribot("What are the main steps involved in rice cultivation?")
    ask_agribot("What are the common diseases that affect that specific crop?")
    ask_agribot("Can you write a Python script for a waste sorting app?")