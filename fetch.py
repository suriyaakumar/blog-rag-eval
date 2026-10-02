from os import environ
from requests import get, RequestException
from frontmatter import loads
from datetime import date, datetime
from dotenv import load_dotenv
from utils import load_json, save_json

load_dotenv()

BLOG_REPO = environ.get("BLOG_REPO", "suriyaakumar/portfolio")
CONTENT_PATH = environ.get("CONTENT_PATH", "src/content/blog")
BRANCH = environ.get("BLOG_BRANCH", "master")

REPO_API_URL = (
    f"https://api.github.com/repos/"
    f"{BLOG_REPO}/contents/{CONTENT_PATH}?ref={BRANCH}"
)

CACHE_FILE = "cache_meta.json"
CONTENT_FILE = "raw_posts.json"

HEADERS = {
    "Authorization": f"token {environ.get('GITHUB_TOKEN', '')}",
    "User-Agent": "blog-rag-eval/0.1 (suriyaakumar personal project)",
}


def get_remote_files():
    try:
        response = get(REPO_API_URL, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return response.json()
    except RequestException as e:
        raise RuntimeError(f"Error fetching remote files: {e}") from e


def parse_posts(raw_text):
    """Split frontmatter (title, date, tags, etc.) from the markdown body."""
    post = loads(raw_text)
    metadata = post.metadata

    for key, value in metadata.items():
        if isinstance(value, (date, datetime)):
            metadata[key] = value.isoformat()

    return {
        "metadata": metadata,
        "content": post.content,
    }


def fetch_post(file_info):
    """Download and parse a single blog post."""
    name = file_info["name"]
    download_url = file_info["download_url"]

    try:
        response = get(download_url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return parse_posts(response.text)
    except RequestException as e:
        raise RuntimeError(f"Error fetching {name}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error parsing {name}: {e}") from e


def sync_posts():
    remote_files = get_remote_files()

    local_cache = load_json(CACHE_FILE, default={})
    local_content = load_json(CONTENT_FILE, default={})

    new_cache = {}
    new_content = {}

    for file_info in remote_files:
        name = file_info["name"]

        if not name.endswith(".md"):
            continue

        remote_sha = file_info["sha"]

        # File is unchanged AND we still have its content locally.
        if (
            name in local_cache
            and local_cache[name] == remote_sha
            and name in local_content
        ):
            print(f"[CACHED] {name} (No changes)")
            new_content[name] = local_content[name]
            new_cache[name] = remote_sha
            continue

        # Either the file changed, or the cache metadata exists but
        # the corresponding content is missing locally.
        if name in local_cache and local_cache[name] == remote_sha:
            print(f"[REFETCH] {name} (Missing from local content)")
        else:
            print(f"[FETCHING] {name} (Changed or new)")

        new_content[name] = fetch_post(file_info)
        new_cache[name] = remote_sha

    save_json(CACHE_FILE, new_cache)
    save_json(CONTENT_FILE, new_content)

    print(f"\nWrote {len(new_content)} posts to {CONTENT_FILE}")


if __name__ == "__main__":
    sync_posts()
