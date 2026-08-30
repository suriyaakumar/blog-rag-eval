import os
import json
import tempfile

def load_json(path, default=None):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return default if default is not None else {}

def save_json(path, data):
    dir_ = os.path.dirname(os.path.abspath(path)) or "."
    # make a temporary file in the same directory to ensure atomic write
    # atomic write ensures that the file is either fully written or not written at all, preventing partial writes
    with tempfile.NamedTemporaryFile("w", dir=dir_, delete=False, suffix=".tmp") as tmp:
        json.dump(data, tmp, indent=2)
        tmp_path = tmp.name
    os.replace(tmp_path, path)
