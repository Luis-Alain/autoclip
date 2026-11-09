"""ASCII-safe slugs for file and folder names."""

from __future__ import annotations

import re
import unicodedata


def slugify(text: str, *, max_length: int = 50, fallback: str = "clip") -> str:
    """Turn any text into a lowercase, ASCII-only, dash-separated slug.

    "¡El GRITO de Ñandú! 😂" -> "el-grito-de-nandu"

    NFKD normalisation splits accented letters into base letter + accent mark
    ("é" -> "e" + "´"), and encoding to ASCII with errors="ignore" then drops the
    marks, emoji, and any script with no ASCII form (Arabic, CJK). If nothing
    survives, ``fallback`` is returned so a file never gets an empty name.
    """
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    slug = slug[:max_length].rstrip("-")
    return slug or fallback
