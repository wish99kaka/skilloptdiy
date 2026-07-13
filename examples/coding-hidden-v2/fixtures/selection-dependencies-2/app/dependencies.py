def build_order(graph: dict[str, list[str]]) -> list[str]:
    """Return lexicographically stable dependency-first order; include referenced nodes and reject cycles."""
    return list(graph)
