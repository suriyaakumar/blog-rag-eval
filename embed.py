from os import environ
from hashlib import sha256
from google import genai
from dotenv import load_dotenv
from utils import load_json, save_json
from time import sleep
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

load_dotenv()
client = genai.Client(api_key=environ.get("GEMINI_API_KEY"))

def hash_chunks(chunk):
    """Generate a SHA-256 hash for a chunk of text."""
    return sha256(chunk.encode("utf-8")).hexdigest()

def embed_content():
    chunks = load_json("chunks.json")
    existing = load_json("embeddings.json", default=[])

     # build a lookup: text_hash -> already-embedded chunk
    existing_by_hash = {hash_chunks(c["text"]): c for c in existing}

    new_chunks = []
    reused = 0
    embedded = 0

    for chunk in chunks:
        hash_chunk = hash_chunks(chunk["text"])

        if hash_chunk in existing_by_hash:
            # reuse the existing embedding
            chunk["embedding"] = existing_by_hash[hash_chunk]["embedding"]
            reused += 1
        else:
            try:
                result = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=chunk["text"],
                )
                chunk["embedding"] = result.embeddings[0].values
                embedded += 1
                print(f"[{embedded+reused}/{len(chunks)}] {chunk['post_slug']}")
                sleep(1)
            except (ResourceExhausted, ServiceUnavailable) as e:
                print(f"Error embedding chunk: {e}")
                sleep(30)
                result = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=chunk["text"],
                )
                chunk["embedding"] = result.embeddings[0].values
                embedded += 1

        new_chunks.append(chunk)
        save_json("embeddings.json", new_chunks)

    print(f"Wrote {len(new_chunks)} embeddings to embeddings.json")
    print(f"Reused: {reused}, Embedded: {embedded}")

if __name__ == "__main__":
    embed_content()
