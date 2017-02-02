import unittest

import pandas as pd

from balsa.cheval.scope.expressions import Expression
from balsa.cheval.scope import SimpleUsage, AttributedUsage, DictLiteral, LinkedFrameUsage, UnsupportedSyntaxError


class TestExpressionParsing(unittest.TestCase):

    def test_simple_usage(self):
        expr = Expression("a + b")

        assert 'a' in expr._symbols
        assert 'b' in expr._symbols

        assert all(isinstance(use, SimpleUsage) for use in expr._symbols['a'])
        assert all(isinstance(use, SimpleUsage) for use in expr._symbols['b'])

        expr = Expression("a")
        assert 'a' in expr._symbols

    def test_attributed_usage(self):
        expr = Expression("a.b")

        assert 'a' in expr._symbols
        assert len(expr._symbols['a']) == 1
        assert isinstance(expr._symbols['a'][0], AttributedUsage)
        assert expr._symbols['a'][0].attribute == 'b'

    def test_dict_literal(self):
        expr = Expression("{a: 1, b: 2, c: 3}")

        assert '__dict0' in expr._symbols
        assert isinstance(expr._symbols['__dict0'], DictLiteral)
        assert isinstance(expr._symbols['__dict0'].series, pd.Series)

    def test_top_level_func(self):
        expr = Expression('log(a.b)')

        assert expr._parsed_expr.startswith('log')

    def test_linked_usage(self):
        expr1 = Expression("a.b.c")

        assert 'a' in expr1._symbols
        assert len(expr1._symbols['a']) == 1
        assert isinstance(expr1._symbols['a'][0], LinkedFrameUsage)
        assert len(expr1._symbols['a'][0].stack) == 2

        expr2 = Expression("a.b.sum(c > d)")

        assert 'a' in expr2._symbols
        assert len(expr2._symbols['a']) == 1
        usage = expr2._symbols['a'][0]
        assert isinstance(usage, LinkedFrameUsage)
        assert usage.func == 'sum'
        assert usage.func_expr == "(c > d)"

    def test_rejected_syntax(self):

        with self.assertRaises(UnsupportedSyntaxError) as context:
            Expression("[a + 1 for a in range(1,6)]")  # List comprehension
            Expression("{a: a + 1 for a in range(5)}")  # Dict comprehension
            Expression("{1,2,3}")  # Set literal
            Expression("q + a if 1 == 2 else b")  # Block If/Else
            Expression("a = b")  # Store
            Expression("del a")  # Del
            Expression("a[1]")  # Subscript
            Expression("a[0:5]")  # Subscript w slice
            Expression("a + {")  # Malformed syntax

if __name__ == '__main__':
    unittest.main()
