def parse_money(value: str) -> float:
    return float(value.strip().replace("$", ""))
