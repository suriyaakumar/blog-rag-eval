from utils import load_json
from dotenv import load_dotenv
from google import genai
from os import environ
import numpy as np

load_dotenv()
client = genai.Client(api_key=environ.get("GEMINI_API_KEY"))

# cosine similarity function to compare two vectors
# used in retrieve.py to find the most similar chunk of text to the user's question
def cosine_similarity(vec_a, vec_b):
    a = np.array(vec_a)
    b = np.array(vec_b)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


def get_top_chunks(question, top_k=1):
    embeddings = load_json("embeddings.json")
    embed_question = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question,
    ).embeddings[0].values

    scores = []

    for chunk in embeddings:
        score = cosine_similarity(embed_question, chunk["embedding"])
        scores.append((score, chunk))

    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:top_k]  # Get topmost k similar chunks since chunk count is less. will increase as blog posts increase XD

def retrieve_content():
    question = input("Enter your question: ")
    top_chunks = get_top_chunks(question, 1)
    for score, chunk in top_chunks:
        print(f"Score: {score:.4f}, Post Slug: {chunk['post_slug']}, Text: {chunk['text'][:100]}...")

if __name__ == "__main__":
    retrieve_content()
