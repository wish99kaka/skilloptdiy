def safe_get(data: dict, path: str, default=None):
    return data.get(path, default)
