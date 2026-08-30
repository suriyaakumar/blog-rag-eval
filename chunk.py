import json
import re

CONTENT_FILE = "raw_posts.json"
CHUNKS_FILE = "chunks.json"

# Matches markdown headings: ## Heading, ### Heading, etc.
HEADING_RE = re.compile(r"^(#{2,4})\s+(.*)$", re.MULTILINE)

MAX_CHUNK_CHARS = 1500  # fallback split size if a single section is too long


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def split_by_headings(markdown_text):
    """Split markdown into (heading, section_text) pairs.
    Content before the first heading is kept under heading=None."""
    matches = list(HEADING_RE.finditer(markdown_text))

   ### if there are no headings, return the whole text as a single section with heading=None
    if not matches:
        return [(None, markdown_text.strip())]

    sections = []

    # if there is content before the first heading, add it as a section with heading=None
    if matches[0].start() > 0:
        preamble = markdown_text[: matches[0].start()].strip()
        if preamble:
            sections.append((None, preamble))

    for i, match in enumerate(matches):
        # get the second part of the header text
        # if the markdown header is "## Heading", match.group(2) will be "Heading"
        heading = match.group(2).strip()
        # start of the section is the end of the current heading,
        start = match.end()
        # and the end is the start of the next heading (or end of text)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
        # get the section text and strip leading/trailing whitespace
        section_text = markdown_text[start:end].strip()
        if section_text:
            sections.append((heading, section_text))

    return sections

"""If a section under one heading is still too long, split on paragraph
    breaks (never mid-sentence) so no single chunk blows past max_chars."""
def split_long_section(text, max_chars=MAX_CHUNK_CHARS):
    # If the text is already under the limit, return it as a single chunk.
    if len(text) <= max_chars:
        return [text]

    # Split the text into paragraphs and build chunks that respect the max_chars limit.
    paragraphs = text.split("\n\n")
    chunks, current = [], ""

    # Build chunks by adding paragraphs until the max_chars limit is reached.
    for para in paragraphs:
        # if the current chunk plus the next paragraph plus two characters for adding blank new line ("\n\n")
        # is under the limit, add it to the current chunk
        if len(current) + len(para) + 2 <= max_chars:
            # if current is not empty, add a blank line before adding the next paragraph
            # otherwise, just add the paragraph to the current chunk
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            current = para
    # add the last chunk if there is any remaining text
    if current:
        chunks.append(current)
    return chunks


def chunk_post(slug, post):
    metadata = post["metadata"]
    content = post["content"]
    title = metadata.get("title", slug)

    # Split the post into sections based on headings, then split long sections into smaller chunks.
    sections = split_by_headings(content)
    chunks = []

    for heading, section_text in sections:
        for piece in split_long_section(section_text):
            chunks.append({
                "post_slug": slug,
                "post_title": title,
                "heading": heading,
                "text": piece,
            })

    return chunks


def chunk_all_posts():
    posts = load_json(CONTENT_FILE)
    all_chunks = []

    for slug, post in posts.items():
        post_chunks = chunk_post(slug, post)
        all_chunks.extend(post_chunks)
        print(f"[CHUNKED] {slug}: {len(post_chunks)} chunks")

    save_json(CHUNKS_FILE, all_chunks)
    print(f"\nWrote {len(all_chunks)} total chunks to {CHUNKS_FILE}")


if __name__ == "__main__":
    chunk_all_posts()
