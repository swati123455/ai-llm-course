from rich import prompt
from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
)
user_input = input("Ask something:")
prompt = f"""
            You are a helpful assistant.
            Answer clearly and simple.
            Question:{user_input}"""

response = generator(prompt)

print(response[0]["generated_text"])