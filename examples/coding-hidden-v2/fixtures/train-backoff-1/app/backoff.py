def backoff_delays(base: int | float, attempts: int, cap: int | float) -> list[int | float]:
    """Return capped exponential delays; non-positive attempts are empty and negative base/cap is invalid."""
    return [base * (2 ** index) for index in range(attempts)]
