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

    special_chars = set(r" .,<>/?;:'|[{]}=+-)(*&^%$#@!`~" + '"')
    regex_chars = set(r"]")
    pyspecchar = list(special_chars - regex_chars)
    escaped_chars = ["\%s" % c for c in regex_chars]
    insertion = ''.join(pyspecchar + escaped_chars)
    unpythonic_regex = r"^\d|[%s\s]+" % insertion

    return not re.match(unpythonic_regex, name) or name in kwlist