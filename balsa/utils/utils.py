from __future__ import division, absolute_import, print_function, unicode_literals

import six
import re
import tokenize
from keyword import kwlist

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
