from retrieve import get_top_chunks
from dotenv import load_dotenv
from google import genai
from os import environ

load_dotenv()
client = genai.Client(api_key=environ.get("GEMINI_API_KEY"))


def build_prompt(question, chunks):
    context = "\n\n".join(
        f"From '{chunk['post_title']}':\n{chunk['text']}"
        for score, chunk in chunks
    )
    return f"""Answer the question using only the context below. If the context doesn't contain the answer, say so.

Context:
{context}

Question: {question}

Answer:"""


def generate_answer():
    question = input("Enter your question: ")
    top_chunks = get_top_chunks(question, top_k=1)

    prompt = build_prompt(question, top_chunks)

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
    )

    print(f"\nAnswer:\n{response.text}")

    print("\nSources:")
    for score, chunk in top_chunks:
        print(f"  - {chunk['post_title']} (score: {score:.4f})")


if __name__ == "__main__":
    generate_answer()
