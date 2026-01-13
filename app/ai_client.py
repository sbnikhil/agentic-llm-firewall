import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_ai_response(history: list):
    """Sends the FULL conversation history to Groq."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Do not mention that data was redacted."}
    ]
    
    messages.extend(history)
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages 
    )
    return completion.choices[0].message.content
