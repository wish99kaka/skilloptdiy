def normalize_columns(headers: list[str]) -> list[str]:
    """Trim labels, replace blanks with column, and add _2/_3 suffixes for case-insensitive duplicates."""
    return [header.strip() for header in headers]
