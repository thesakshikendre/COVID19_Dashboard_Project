"""
utils.py - Helper utility functions
"""
import json
from pathlib import Path

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

if __name__ == "__main__":
    print("Utils Module Loaded")