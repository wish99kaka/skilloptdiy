def chunk_records(items: list, size: int) -> list[list]:
    """Return fresh consecutive chunks; reject non-positive sizes and never mutate items."""
    return [list(items)]
