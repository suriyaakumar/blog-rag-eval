# Blog RAG Eval

A Python retrieval-augmented generation (RAG) project for answering questions about a personal blog. It fetches Markdown posts from GitHub, builds Gemini embeddings, and searches them with NumPy cosine similarity. The index is a JSON file; no vector database is required.

The repository includes an indexing pipeline, interactive retrieval and answer scripts, an AWS Lambda handler with conversation support, and an automated evaluation runner with a Gemini judge. It does not include a frontend, deployment infrastructure, or CI workflow.

## How it works

```text
GitHub Markdown posts
        |
     fetch.py  -> raw_posts.json + cache_meta.json
        |
     chunk.py  -> chunks.json
        |
     embed.py  -> embeddings.json
        |
        +-- retrieve.py / generate.py  -> local question and answer
        |
        +-- upload to S3 -> lambda_handler.py -> HTTP question and answer
                                    ^
                                    |
                             eval/run_eval.py
                                    |
                             eval/scorer.py -> eval/reports/*.json
```

1. **Fetch:** Read lowercase `.md` files directly in the configured GitHub directory, parse YAML frontmatter, and reuse unchanged posts using GitHub file SHAs. Deleted posts are removed from the synchronized outputs.
2. **Chunk:** Split at `##` through `####` headings, then group paragraphs toward a 1,500-character target. Long individual paragraphs can exceed the target. Each chunk retains its source filename, title, and heading.
3. **Embed:** Embed chunk text with `gemini-embedding-001` and the `RETRIEVAL_DOCUMENT` task type. Reuse existing vectors when the text has the same SHA-256 hash.
4. **Retrieve:** Embed the question with `RETRIEVAL_QUERY`, score every chunk by cosine similarity, and keep scores at or above `0.60`.
5. **Answer:** Send retrieved passages to `gemini-3.1-flash-lite`. Prompts instruct the model to use blog evidence, reject false premises, acknowledge missing information, and ignore instructions embedded in questions.

The local CLIs use one qualifying chunk. Lambda uses up to three and can rewrite a follow-up into a standalone retrieval query using client-supplied conversation history.

## Local setup

Use a Python version compatible with the pinned dependencies in [`requirements.txt`](requirements.txt). The repository does not declare a supported Python version.

Run commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

Create a `.env` file:

```dotenv
GEMINI_API_KEY=your_gemini_api_key
GITHUB_TOKEN=your_github_token

# Optional source settings; these are the defaults.
BLOG_REPO=suriyaakumar/portfolio
CONTENT_PATH=src/content/blog
BLOG_BRANCH=master
```

The local fetch, embedding, retrieval, generation, and evaluation modules load `.env` with `python-dotenv`. Lambda reads its environment directly. Clients and configuration are initialized at import time, so configure variables before starting a script.

`fetch.py` always sends a GitHub authorization header, even when the token is empty. Configure a usable token with access to the source repository.

### Build or refresh the index

```bash
python fetch.py
python chunk.py
python embed.py
```

Run the stages sequentially after blog updates. They do not invoke one another, and their data paths are relative to the current working directory.

| Generated file | Contents |
| --- | --- |
| `cache_meta.json` | Source filenames mapped to GitHub file SHAs. |
| `raw_posts.json` | Parsed metadata and Markdown bodies, keyed by filename. |
| `chunks.json` | Text chunks with source metadata. |
| `embeddings.json` | Chunks augmented with numeric embedding vectors. |

Fetching and embedding make external API requests. Unchanged posts and previously embedded text are reused. New embeddings normally include a one-second pause between calls. `embed.py` saves the completed index prefix after every chunk using atomic file replacement.

The generated data files, `.env`, virtual environments, and Lambda build artifacts are ignored by Git.

### Ask a question

Generate an answer:

```bash
python generate.py
```

Or inspect retrieval alone:

```bash
python retrieve.py
```

Enter a question at the prompt, for example:

```text
Why did the author choose Astro for their portfolio?
```

Each script accepts one question and exits. `generate.py` prints an answer and source titles with similarity scores; `retrieve.py` prints the score, filename, and a passage preview. The local CLIs have no conversation history or chat loop. Local generation still calls the model when retrieval returns no qualifying chunks, relying on the prompt to abstain.

## AWS Lambda

[`lambda_handler.py`](lambda_handler.py) exposes the entry point `lambda_handler.lambda_handler`. It loads the index from S3 for each accepted question and returns an HTTP-style response with a JSON body.

### Deployment setup

1. Build the index locally and upload `embeddings.json` to S3 under the exact key `embeddings.json`.
2. Package `lambda_handler.py` and its dependencies for your Lambda Python runtime and architecture. The handler imports `boto3`, which is absent from `requirements.txt`; install it separately for local use or include it in the deployment package as needed.
3. Give the execution role `s3:GetObject` access to the index and provide connectivity to S3 and the Gemini API.
4. Set the Lambda environment variables below.
5. Connect an HTTP endpoint that supplies request headers and a JSON-string `body` in the event.

| Variable | Purpose |
| --- | --- |
| `GEMINI_API_KEY` | Key for question embeddings, query rewriting, and answer generation. |
| `EMBEDDINGS_BUCKET` | Bucket containing `embeddings.json`. |
| `APP_SECRET` | Nonempty shared secret required in the request header. |

Rebuilding the local index does not refresh S3. For example, with AWS CLI credentials and a shell variable set to your bucket:

```bash
aws s3 cp embeddings.json "s3://$EMBEDDINGS_BUCKET/embeddings.json"
```

Python dotenv loading does not export variables to your shell; set shell variables used in commands separately.

### Request and response

Set `BLOG_RAG_ENDPOINT` and `APP_SECRET` in your shell, then send:

```bash
curl -X POST "$BLOG_RAG_ENDPOINT" \
  -H "Content-Type: application/json" \
  -H "x-app-secret: $APP_SECRET" \
  -d '{"question":"Why did the author choose Astro for their portfolio?"}'
```

The body accepts a required `question` string of at most 500 characters and optional `history`:

```json
{
  "question": "What other reasons did the author give?",
  "history": [
    {
      "question": "Why did the author choose Astro for their portfolio?",
      "answer": "The author chose Astro for its zero-JavaScript-by-default approach."
    }
  ]
}
```

The history above illustrates the request format. The client must retain and send previous turns; Lambda does not store sessions. History is sanitized as follows:

- A non-list value becomes an empty history.
- Only the last four input entries are considered; non-dictionary entries are skipped.
- Each question and answer is converted to a string and truncated to 1,500 characters; both must be nonempty to retain the turn.

With valid history, Lambda asks `gemini-3.5-flash-lite` to rewrite the current question for retrieval, with a 100-token output limit and `minimal` thinking level. The rewritten query is capped at 500 characters. A rewrite `ServerError` falls back to the original question. Answer generation receives the original question, sanitized history, and retrieved passages. The prompt treats history as conversational context and requires factual claims to come from the retrieved blog text.

A successful answer body looks like this; values are illustrative:

```json
{
  "answer": "Generated answer based on the retrieved blog content.",
  "sources": [
    {"title": "This home I claim", "score": 0.82}
  ]
}
```

Sources are returned per chunk, so multiple chunks can produce duplicate titles. Scores are cosine similarities, not calibrated probabilities.

| Status | Behavior |
| --- | --- |
| `200` | Generated answer and up to three qualifying source entries. |
| `403` | `APP_SECRET` is unset/empty or the supplied secret does not match. |
| `400` | Question is missing/empty or exceeds 500 characters. |
| `500` | A generation `ServerError` returns an empty answer and sources, with `message: "MEMORY is temporarily unavailable. Please try again."`. |

The handler recognizes `x-app-secret` and `X-App-Secret`. Malformed JSON, unexpected request types, S3 failures, embedding errors, and most other upstream failures are not explicitly handled. CORS headers and preflight handling are not implemented.

**Known response bug:** When no chunks qualify, the handler skips answer generation and constructs a `200` response containing `"The blog doesn't cover that."` and an empty source list. A trailing comma currently makes that return value a one-element tuple instead of the expected response dictionary, which can cause an invalid HTTP integration response.

Lambda logs `[LATENCY]` timings for query embedding, similarity search, query rewriting, prompt construction, answer generation, and total handler time. It does not cache the S3 index between requests.

## Automated evaluation

[`eval/run_eval.py`](eval/run_eval.py) sends questions to a deployed endpoint, checks returned source titles, and grades answers using [`eval/scorer.py`](eval/scorer.py).

Add these variables to `.env`:

```dotenv
EVAL_ENDPOINT=https://your-api-endpoint
EVAL_APP_SECRET=the_same_value_as_lambdas_app_secret
# Optional judge override; this is the default.
JUDGE_MODEL=gemini-3.1-flash-lite
```

Evaluation also requires `GEMINI_API_KEY` for the judge. Run a judge smoke check or the full evaluation from the repository root:

```bash
# Grades two example answers; this makes Gemini calls.
python eval/scorer.py

# The optional argument labels the report; the default label is "run".
python eval/run_eval.py baseline
```

The full run calls both the deployed service and the judge, consuming their API quota. The runner pauses three seconds after each case and allows up to three HTTP attempts for request exceptions or statuses `429`, `500`, `502`, `503`, and `504`, with exponential delays and a 35-second timeout per attempt. Other non-200 statuses are recorded without retrying. The judge allows two attempts; exhausted judge failures are recorded as unjudged.

### Test set and scoring

[`eval/testset.json`](eval/testset.json) currently has 15 cases, with three in each category:

| Category | Expected behavior |
| --- | --- |
| `answerable` | Answer from the blog and retrieve the expected source. |
| `unanswerable_related` | Acknowledge that the blog does not provide the specific information. |
| `false_premise` | Reject or correct an unsupported premise. |
| `unrelated` | Decline questions outside the blog's scope. |
| `injection` | Ignore attempts to override the blog-only instructions. |

Each case contains `id`, `question`, `category`, and `expected_source` (a post title or `null`). The runner and scorer also support optional `history` and `expected_behavior` fields, and the scorer has guidance for a `followup` category. The current test set contains no follow-up cases. History is sent as a fixture; the runner does not build conversations from preceding test results.

Two checks are recorded independently:

- **Source match:** Exact, case-sensitive membership of the expected title in returned sources. A `null` expected source means the check is not applicable, even if sources are returned.
- **Judge verdict:** Gemini returns a pass/fail verdict and a short reason based on the category, question, answer, and optional history/expectation. The judge receives neither blog passages nor returned sources, so it cannot independently verify factual grounding.

The console summary shows passes, failures, unjudged cases, request errors, source hits, and top-score ranges by category. Overall passes count judge passes only; source matches do not determine that total, and errors/unjudged cases remain in its denominator.

Reports are saved to `eval/reports/YYYYMMDD-HHMMSS-<label>.json`, with `label`, per-category `summary`, and per-case `results`. Successful cases include the answer, source titles, top similarity score, source match, and judge verdict/reason; request failures include an error. Reports use the execution environment's local timestamp. `eval/reports/` is not currently excluded by `.gitignore`.

This is a small diagnostic set with no reference answers or checked-in benchmark reports. Review judge decisions against the blog, especially when assessing factual correctness or changing retrieval settings.

## Configuration reference

### Environment variables

| Variable | Used by | Default |
| --- | --- | --- |
| `GEMINI_API_KEY` | Embedding, retrieval, generation, Lambda, evaluation judge | No application default |
| `GITHUB_TOKEN` | Fetcher | Empty string; configure a valid token |
| `BLOG_REPO` | Fetcher | `suriyaakumar/portfolio` |
| `CONTENT_PATH` | Fetcher | `src/content/blog` |
| `BLOG_BRANCH` | Fetcher | `master` |
| `EMBEDDINGS_BUCKET` | Lambda | Unset |
| `APP_SECRET` | Lambda | Unset; requests are rejected |
| `EVAL_ENDPOINT` | Evaluation runner | Unset |
| `EVAL_APP_SECRET` | Evaluation runner | Unset |
| `JUDGE_MODEL` | Evaluation judge | `gemini-3.1-flash-lite` |

### Settings in source

| Setting | Current value | Location |
| --- | --- | --- |
| Chunk target | 1,500 characters | `chunk.py`: `MAX_CHUNK_CHARS` |
| Embedding model | `gemini-embedding-001` | `embed.py`, `retrieve.py`, `lambda_handler.py` |
| Answer model | `gemini-3.1-flash-lite` | `generate.py`; `lambda_handler.py`: `GEN_MODEL` |
| Query rewrite model | `gemini-3.5-flash-lite` | `lambda_handler.py`: `REWRITE_MODEL` |
| Rewrite output/thinking | 100 tokens / `minimal` | `lambda_handler.py`: `rewriteConfig` |
| Similarity cutoff | `0.60`, inclusive | `retrieve.py`, `lambda_handler.py`: `SIMILARITY_THRESHOLD` |
| Retrieval function default | `top_k=3` | `retrieve.py`, `lambda_handler.py` |
| CLI retrieval count | One chunk | Explicit calls in `retrieve.py`, `generate.py` |
| Lambda retrieval count | Up to three chunks | Uses `get_top_chunks()` default |
| Question limit | 500 characters | `lambda_handler.py`: `MAX_QUESTION_LENGTH` |
| History limits | Last four entries; 1,500 characters per question/answer | `lambda_handler.py`: `MAX_HISTORY_TURNS`, `MAX_HISTORY_CHARS` |
| S3 object key | `embeddings.json` | `lambda_handler.py`: `KEY` |
| Evaluation pause | Three seconds per case | `eval/run_eval.py`: `DELAY_SECONDS` |
| Judge configuration | JSON response, temperature `0` | `eval/scorer.py` |

Model names are the values requested by the code; availability depends on your Gemini account and API environment. Local and Lambda retrieval/generation are separate implementations, so update and evaluate both when changing shared behavior.

## Index format

`embeddings.json` is an array of chunk objects with vectors:

```json
[
  {
    "post_slug": "example.md",
    "post_title": "Example post",
    "heading": "Retrieval",
    "text": "Section body.",
    "embedding": [0.12, -0.03, 0.08]
  }
]
```

The vector above is illustrative; real vectors use the model's full output dimension. `post_slug` is the source filename, and `heading` is `null` for a preamble or heading-free post. Titles fall back to the filename when frontmatter has no `title` key. Only chunk `text` is embedded; titles and headings are not included in embedding input.

The index has no model/version manifest, stable chunk IDs, or source URLs. Keep document and query embedding models compatible and rebuild vectors when changing the embedding model or configuration. Reuse is keyed only by text, so existing vectors are otherwise retained across such changes.

## Repository files

| File | Responsibility |
| --- | --- |
| [`fetch.py`](fetch.py) | Synchronize Markdown posts from GitHub and parse frontmatter. |
| [`chunk.py`](chunk.py) | Split posts into sections and paragraph groups. |
| [`embed.py`](embed.py) | Generate, reuse, and checkpoint document embeddings. |
| [`retrieve.py`](retrieve.py) | Search the local index and expose an interactive retrieval CLI. |
| [`generate.py`](generate.py) | Generate a local answer from retrieved context. |
| [`lambda_handler.py`](lambda_handler.py) | Authenticate requests, load S3 data, rewrite follow-ups, retrieve, and generate answers. |
| [`utils.py`](utils.py) | Load JSON and save it with atomic file replacement. |
| [`eval/run_eval.py`](eval/run_eval.py) | Call the deployed endpoint, summarize scores, and save reports. |
| [`eval/scorer.py`](eval/scorer.py) | Check source titles and judge answer behavior with Gemini. |
| [`eval/testset.json`](eval/testset.json) | Evaluation questions and expected source titles. |
| [`requirements.txt`](requirements.txt) | Pinned Python dependencies; excludes `boto3`. |

## Limitations and troubleshooting

- **Index completeness:** An interrupted embedding run can leave a valid but partial index. A zero-chunk embedding run does not clear an existing index. Finish indexing before querying or uploading to S3.
- **Embedding retries:** `embed.py` catches only `google.api_core` `ResourceExhausted` and `ServiceUnavailable`, waits 30 seconds, and retries once. Other SDK error types propagate, and the retry omits `RETRIEVAL_DOCUMENT` configuration.
- **Corpus changes:** Fetch caches do not record repository/branch/path identity. Start with fresh fetch caches when switching source repositories or branches.
- **Retrieval coverage:** There is no lexical search, reranking, or deduplication. Inspect retrieved passages and the cutoff when an answerable question gets an abstention. Heading matching uses a regex and can split heading-like lines inside fenced code.
- **Grounding:** Answer support and injection resistance rely on prompts; no post-generation evidence validator is implemented. The automated judge also lacks source text.
- **HTTP failures:** Check the shared secret for `403`, question length for `400`, and bucket/key/role permissions for S3 errors. The empty-retrieval response has the tuple bug described above. Other request and upstream errors can propagate.
- **Local/deployed differences:** The CLI uses one chunk and no history, while Lambda uses up to three and supports follow-ups. Check both code paths and the S3 index revision when answers differ.
- **File-based operation:** Stages and uploads are separate operations without a shared transaction or concurrent-writer coordination. There is no question/answer cache, scheduler, or application-level rate limit.
