import json
import os
import boto3
import numpy as np
from google import genai
from google.genai import types
from google.genai.errors import ServerError
from time import perf_counter

s3 = boto3.client("s3")
BUCKET = os.environ.get("EMBEDDINGS_BUCKET")
APP_SECRET = os.environ.get("APP_SECRET")
KEY = "embeddings.json"

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
rewriteConfig = types.GenerateContentConfig(
    max_output_tokens=100,
    thinking_config=types.ThinkingConfig(
        thinking_level="minimal",
    ),
)

MAX_QUESTION_LENGTH = 500
SIMILARITY_THRESHOLD = 0.60  # tune this against your own test set
MAX_HISTORY_TURNS = 4
MAX_HISTORY_CHARS = 1500
REWRITE_MODEL = "gemini-3.5-flash-lite"  # can be a smaller model if you want to save cost


def load_embeddings_from_s3():
    response = s3.get_object(Bucket=BUCKET, Key=KEY)
    return json.loads(response["Body"].read())


def cosine_similarity(vec_a, vec_b):
    a = np.array(vec_a)
    b = np.array(vec_b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_top_chunks(question, embeddings, top_k=3):
    t = perf_counter()
    embed_question = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question,
        config={"task_type": "RETRIEVAL_QUERY"},
    ).embeddings[0].values
    embedding_ms = (perf_counter() - t) * 1000

    t = perf_counter()
    scores = []
    for chunk in embeddings:
        score = cosine_similarity(embed_question, chunk["embedding"])
        if score >= SIMILARITY_THRESHOLD:
            scores.append((score, chunk))

    scores.sort(key=lambda x: x[0], reverse=True)
    search_ms = (perf_counter() - t) * 1000

    print(f"[LATENCY] embedding={embedding_ms:.0f}ms retrieval={search_ms:.0f}ms")

    return scores[:top_k]

def sanitize_history(raw):
    # client-supplied, so treat as untrusted: validate shape, cap turns and sizes
    if not isinstance(raw, list):
        return []
    clean = []
    for turn in raw[-MAX_HISTORY_TURNS:]:
        if not isinstance(turn, dict):
            continue
        q = str(turn.get("question", ""))[:MAX_HISTORY_CHARS]
        a = str(turn.get("answer", ""))[:MAX_HISTORY_CHARS]
        if q and a:
            clean.append({"question": q, "answer": a})
    return clean


def format_history(history):
    return "\n".join(f"User: {t['question']}\nAssistant: {t['answer']}" for t in history)


def rewrite_query(question, history):
    if not history:
        return question
    prompt = f"""Rewrite the follow-up question as a standalone question that makes sense without the conversation. If it is already standalone, return it unchanged. Return only the question.

Conversation:
{format_history(history)}

Follow-up: {question}

Standalone question:"""
    try:
        response = client.models.generate_content(model=REWRITE_MODEL, contents=prompt, config=rewriteConfig)
        return (response.text or "").strip()[:MAX_QUESTION_LENGTH] or question
    except ServerError as e:
        print(f"Error rewriting query: {e}")
        return question

def lambda_handler(event, context):
    total_start = perf_counter()
    headers = event.get("headers", {})
    provided_secret = headers.get("x-app-secret") or headers.get("X-App-Secret")

    if not APP_SECRET or provided_secret != APP_SECRET:
        return {
            "headers": {
                "Content-Type": "application/json",
            },
            "statusCode": 403,
            "body": json.dumps({"error": "Forbidden"}),
        }

    body = json.loads(event.get("body", "{}"))
    question = body.get("question")

    if not isinstance(question,str) or not question:
        return {
            "headers": {
                "Content-Type": "application/json",
            },
            "statusCode": 400,
            "body": json.dumps({"error": "Missing 'question' in request body"}),
        }

    if len(question) > MAX_QUESTION_LENGTH:
        return {
            "headers": {
                "Content-Type": "application/json",
            },
            "statusCode": 400,
            "body": json.dumps({"error": f"Question too long (max {MAX_QUESTION_LENGTH} characters)"}),
        }

    embeddings = load_embeddings_from_s3()

    history = sanitize_history(body.get("history"))

    t = perf_counter()
    standalone = rewrite_query(question, history)
    rewrite_ms = (perf_counter() - t) * 1000

    t = perf_counter()
    top_chunks = get_top_chunks(standalone, embeddings)
    retrieval_ms = (perf_counter() - t) * 1000

    # If no relevant chunks are found, return a default response indicating that the blog doesn't cover the topic.
    # This prevents generating an answer based on unrelated or insufficient context.
    if not top_chunks:
        total_ms = (perf_counter() - total_start) * 1000
        print(
            f"[LATENCY] rewrite={rewrite_ms:.0f}ms "
            f"retrieval={retrieval_ms:.0f}ms "
            f"total={total_ms:.0f}ms"
        )

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "standalone_query": standalone,
                "chunks": [],
            }),
        }

    total_ms = (perf_counter() - total_start) * 1000
    print(
            f"[LATENCY] rewrite={rewrite_ms:.0f}ms "
            f"retrieval={retrieval_ms:.0f}ms "
            f"total={total_ms:.0f}ms"
    )

    return {
    "statusCode": 200,
    "headers": {
        "Content-Type": "application/json",
    },
    "body": json.dumps({
        "standalone_query": standalone,
        "chunks": [
            {
                "post_slug": chunk["post_slug"],
                "post_title": chunk["post_title"],
                "heading": chunk["heading"],
                "text": chunk["text"],
                "score": float(score),
            }
            for score, chunk in top_chunks
        ],
    }),
}
