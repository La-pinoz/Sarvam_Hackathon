# kb/database.py
"""
Knowledge base loader for FasalVachak.

Loads all crop disease JSON files and merges them into a single flat
dictionary (FULL_KB_DICT) keyed by disease_id.

This module uses Path(__file__) to derive absolute paths so it works
correctly regardless of the working directory from which the server is started.
"""

from __future__ import annotations

import json
from pathlib import Path

# Directory that contains this file (i.e. the kb/ folder itself)
_KB_DIR = Path(__file__).resolve().parent

# JSON files to load, resolved to absolute paths
_FILES_TO_LOAD = [
    _KB_DIR / "cotton.json",
    _KB_DIR / "rice.json",
]


def build_master_kb() -> dict:
    """
    Loads and merges all crop JSON files into a single O(1) lookup dictionary.
    Raises a ValueError if duplicate disease IDs are found across files.
    """
    master_kb: dict = {}

    for file_path in _FILES_TO_LOAD:
        with open(file_path, "r", encoding="utf-8") as f:
            crop_data: dict = json.load(f)

        # Guard against accidental ID collisions across crop files
        overlap = set(master_kb) & set(crop_data)
        if overlap:
            raise ValueError(
                f"Duplicate disease_id(s) found across KB files: {overlap}"
            )

        master_kb.update(crop_data)

    return master_kb


# Executed ONCE at import time; stays cached in RAM for the server lifetime.
FULL_KB_DICT: dict = build_master_kb()
