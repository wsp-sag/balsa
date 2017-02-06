import re
from keyword import kwlist


def name_is_pythonic(name):
    """
    Tests that the name is a valid Python variable name and does not collide with reserved keywords

    Args:
        name (str): Name to test

    Returns:
        bool: If the name is 'Pythonic'

    """

    # TODO: Make this work somehow in Python 2

    return name.isidentifier() and name not in kwlist
