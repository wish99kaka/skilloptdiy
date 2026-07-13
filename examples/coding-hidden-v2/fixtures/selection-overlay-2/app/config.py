def overlay_config(base: dict, overrides: dict) -> dict:
    """Return a new mapping where non-None overrides win without mutating either input."""
    result = base
    result.update(overrides)
    return result
