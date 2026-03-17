import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "tinyllama",
        "prompt": "Explain AI in simple terms",
        "stream": False
    }
)

print(response.json()["response"])