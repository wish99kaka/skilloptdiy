def total_amount(csv_text):
    total = 0
    for line in csv_text.splitlines():
        total += float(line.split(",")[0])
    return total

