def parse_query(query):
    result = {}
    for part in query.split("&"):
        key, value = part.split("=")
        result[key] = value
    return result

