from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

llm = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

docs = [
    "AI is the simulation of Human Intelligence.",
    "Machine Learning is the subset of AI,",
    "Deep Learning uses neural networks"
]

doc_vectors = embedder.encode(docs)

index = faiss.IndexFlatL2(len(doc_vectors[0]))
index.add(np.array(doc_vectors))

query = input("Ask something: ")

query_vector = embedder.encode([query])
D, I = index.search(np.array(query_vector), k = 2)

distance = D[0][0]

if distance > 1.5:
    print("I dont know what you mean")
else:
    context = " ".join([docs[i] for i in I[0]])

    prompt = f"""
    You are helpful assistant.
    Answer only from context.
    If answer not found say "I dont know what you mean.".

    Context is {context}
    Question is {query}
    """

    response = llm(prompt, max_new_tokens=100)

    print(response[0]["generated_text"])


