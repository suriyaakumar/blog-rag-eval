import os
import json
import requests
import frontmatter
from datetime import date, datetime
from dotenv import load_dotenv

load_dotenv()

BLOG_REPO = os.environ.get("BLOG_REPO", "suriyaakumar/portfolio")
CONTENT_PATH = os.environ.get("CONTENT_PATH", "src/content/blog")
BRANCH = os.environ.get("GITHUB_REF", "feat/portfolio_v2")
REPO_API_URL = f"https://api.github.com/repos/{BLOG_REPO}/contents/{CONTENT_PATH}?ref={BRANCH}"

CACHE_FILE = "cache_meta.json"
CONTENT_FILE = "raw_posts.json"

HEADERS = {
    "Authorization": f"token {os.environ.get('GITHUB_TOKEN', '')}",
     "User-Agent": "blog-rag-eval/0.1 (suriyaakumar personal project)"
}

def get_remote_files():
    try:
        response = requests.get(REPO_API_URL, headers=HEADERS, timeout=10)
        response.raise_for_status()
        # Returns a list of files with 'name', 'path', 'sha', and 'download_url'
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching remote files: {e}")
        return []

def load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def parse_posts(raw_text):
    """Split frontmatter (title, date, tags, etc.) from the markdown body."""
    post = frontmatter.loads(raw_text)
    metadata = post.metadata
    for key,value in metadata.items():
        if isinstance(value, (date, datetime)):
            metadata[key] = value.isoformat()
    return {
        "metadata": metadata,
        "content": post.content
    }

def sync_posts():
    # get the list of files from the GitHub repo and load local cache and content
    remote_files = get_remote_files()
    local_cache = load_json(CACHE_FILE)
    local_content = load_json(CONTENT_FILE)

    new_cache = {}
    new_content = {}

    # get the files from the GitHub repo and compare with local cache to see if any have changed
    for file_info in remote_files:
        name = file_info["name"]
        if not name.endswith(".md"):
            continue

        remote_sha = file_info["sha"]
        download_url = file_info["download_url"]

        # Check if file is new or modified based on GitHub's SHA
        if name not in local_cache or local_cache[name] != remote_sha:
            print(f"[FETCHING] {name} (Changed or new)")
            try:
                res = requests.get(download_url, headers=HEADERS, timeout=10)
                res.raise_for_status()
            except requests.RequestException as e:
                print(f"Error fetching {name}: {e}")
                # keep old cached version if we have one, rather than losing it
                if name in local_content:
                    new_content[name] = local_content[name]
                    new_cache[name] = local_cache[name]
                continue

            try:
                new_content[name] = parse_posts(res.text)
            except Exception as e:
                print(f"Error parsing {name}: {e}")
                continue

            new_cache[name] = remote_sha
        else:
            print(f"[CACHED] {name} (No changes)")
            new_content[name] = local_content[name]
            new_cache[name] = local_cache[name]

    # Save the updated metadata cache
    save_json(CACHE_FILE, new_cache)
    save_json(CONTENT_FILE, new_content)
    print(f"\nWrote {len(new_content)} posts to {CONTENT_FILE}")

if __name__ == "__main__":
    sync_posts()
