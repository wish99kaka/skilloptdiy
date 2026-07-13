def compare_versions(left, right):
    if left == right:
        return 0
    return 1 if left > right else -1

