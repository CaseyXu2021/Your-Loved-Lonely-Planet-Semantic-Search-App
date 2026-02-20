"""
Image display component for Streamlit.
Cycles through images in book_picture/{region}/ with Prev/Next buttons.
"""

import os
import base64
import streamlit as st
from typing import List, Optional
from pathlib import Path


# =============================================================================
# Path setup (script directory as base for relative paths)
# =============================================================================

_SCRIPT_DIR = Path(__file__).resolve().parent


# =============================================================================
# Image file discovery and utilities
# =============================================================================

def get_image_files(folder_path: str) -> List[str]:
    """Return sorted list of image file paths in folder (jpg, jpeg, png, webp, gif)."""
    if not os.path.exists(folder_path):
        return []

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    image_files = []

    try:
        for file in sorted(os.listdir(folder_path)):
            if Path(file).suffix.lower() in valid_extensions:
                full_path = os.path.join(folder_path, file)
                if os.path.isfile(full_path):
                    image_files.append(full_path)
    except Exception as e:
        st.warning(f"Error reading folder: {str(e)}")
        return []

    return image_files


def image_to_base64(image_path: str) -> Optional[str]:
    """Convert image file to base64-encoded string for embedding."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"Error converting image {image_path}: {str(e)}")
        return None


def get_image_mime_type(image_path: str) -> str:
    """Return MIME type for image based on file extension."""
    ext = Path(image_path).suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    return mime_types.get(ext, "image/jpeg")


# =============================================================================
# Streamlit-native image cycle (current default)
# =============================================================================

def image_carousel(country_name: str, key: Optional[str] = None, auto_play_seconds: int = 0) -> None:
    """
    Display images from book_picture/{country}/ in vertical order (top to bottom).
    """
    folder = _SCRIPT_DIR / "book_picture" / country_name
    paths = get_image_files(str(folder))

    if not paths:
        st.info("No images for this destination.")
        return

    for i, path in enumerate(paths, 1):
        st.image(path, width="stretch")
