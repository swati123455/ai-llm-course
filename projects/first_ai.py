from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

user_input = input("Ask anything:")
response = client.responses.create(
    model="gpt-4.1-mini",
    input=user_input
)

print(response.output_text)