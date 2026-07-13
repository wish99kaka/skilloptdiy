def order_total(items, discount_percent=0, tax_percent=0):
    subtotal = sum(item["price"] for item in items)
    return round(subtotal, 2)

