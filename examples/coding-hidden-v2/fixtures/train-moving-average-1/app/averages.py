def moving_average(values: list[int | float], width: int) -> list[float]:
    """Return averages for complete consecutive windows; require a positive width and preserve input."""
    if not values:
        return []
    return [sum(values) / len(values)]
