def sort_names(names):
    """Return a new list of names in alphabetical order.

    The input list is not modified. Empty input returns an empty list.
    """
    return sorted(names)


def test_sort_names():
    assert sort_names(["b", "a"]) == ["a", "b"]
    assert sort_names([]) == []
