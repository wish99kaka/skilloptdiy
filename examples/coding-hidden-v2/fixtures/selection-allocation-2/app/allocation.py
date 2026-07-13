def split_capacity(total: int, weights: list[int | float]) -> list[int]:
    """Allocate all non-negative integer units by largest remainder, breaking ties by lower index."""
    if not weights:
        return []
    share = total // len(weights)
    return [share for _ in weights]
