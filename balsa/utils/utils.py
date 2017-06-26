from __future__ import division, absolute_import, print_function, unicode_literals

from contextlib import contextmanager

import six
import re
import tokenize
from keyword import kwlist

from six import string_types

try:
    from pathlib import Path
except ImportError:
    Path = None

if six.PY3:
    def is_identifier(name):
        """
        Tests that the name is a valid Python variable name and does not collide with reserved keywords

        Args:
            name (str): Name to test

        Returns:
            bool: If the name is 'Pythonic'

        """

        return name.isidentifier() and name not in kwlist
else:
    def is_identifier(name):
        """
        Tests that the name is a valid Python variable name and does not collide with reserved keywords

        Args:
            name (str): Name to test

        Returns:
            bool: If the name is 'Pythonic'

        """

        return bool(re.match(tokenize.Name + '$', name)) and name not in kwlist


@contextmanager
def open_file(file_handle, **kwargs):
    """
    Context manager for opening files provided as several different types. Supports a file handler as a str, unicode,
    pathlib.Path, or an already-opened handler.

    Args:
        file_handle (str or unicode or Path or File): The item to be opened or is already open.
        **kwargs: Keyword args passed to open. Usually mode='w'.

    Yields:
        File: The opened file handler. Automatically closed once out of context.

    """
    opened = False
    if isinstance(file_handle, string_types):
        f = open(file_handle, **kwargs)
        opened = True
    elif Path is not None and isinstance(file_handle, Path):
        f = file_handle.open(**kwargs)
        opened = True
    else:
        f = file_handle

    try:
        yield f
    finally:
        if opened:
            f.close()
