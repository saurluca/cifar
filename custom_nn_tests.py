import pytest


def add(a, b):
    return a+b

def test_positive_numbers():
    assert add(2, 3) == 5