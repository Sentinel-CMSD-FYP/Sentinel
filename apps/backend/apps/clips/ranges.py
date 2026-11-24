"""HTTP byte-range support, so a browser can seek inside a video.

Django's ``FileResponse`` ignores the ``Range`` header, and without ``206`` replies browsers
cannot seek. Only a single range is supported; anything else is ignored, which HTTP allows.
"""

from __future__ import annotations

import re

# Largest slice sent per request. A player asking for "bytes=0-" just gets the first slice
# and asks for the next one, and memory per request stays bounded.
MAX_CHUNK = 8 * 1024 * 1024

_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")


class RangeNotSatisfiable(Exception):
    pass


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """Inclusive ``(start, end)`` byte positions, or ``None`` to send the whole file.

    Raises ``RangeNotSatisfiable`` when the range starts past the end of the file.
    """
    if not header:
        return None
    m = _RANGE.match(header.strip())
    if not m or (m[1] == "" and m[2] == ""):
        return None  # malformed or multi-range: ignore the header

    if m[1] == "":  # "bytes=-N": the last N bytes
        n = int(m[2])
        if n == 0:
            raise RangeNotSatisfiable
        start, end = max(size - n, 0), size - 1
    else:
        start = int(m[1])
        end = int(m[2]) if m[2] else size - 1
    if start >= size or (m[2] and end < start):
        raise RangeNotSatisfiable
    return start, min(end, size - 1, start + MAX_CHUNK - 1)
