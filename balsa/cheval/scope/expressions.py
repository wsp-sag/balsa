from __future__ import division, absolute_import, print_function, unicode_literals

from balsa.cheval.scope.parsing import ExpressionProcessor, SimpleUsage, DictLiteral, AttributedUsage, LinkedFrameUsage
from six import iteritems, string_types


class InconsistentUsageError(RuntimeError):
    pass


class Expression(object):

    def __init__(self, expr):
        self._raw_expr = expr
        parsed, symbols = ExpressionProcessor.parse(expr)
        self._parsed_expr = parsed
        self._symbols = symbols

    def symbols(self):
        for item in iteritems(self._symbols): yield item


class ExpressionContainer(object):

    def __init__(self, model):
        self._expressions = []
        self._model_ref = model
        self._modified = True
        self._cached_types = None
        self._cached_literals = None

    def __iter__(self):
        for expr in self._expressions: yield expr

    def __len__(self):
        return len(self._expressions)

    def append(self, expression_or_iterable):
        """
        Appends an expression into the LogitModel scope. Expressions are assumed to be order-dependant, just in case.

        If the given argument is an iterable of strings, then this method will perform a "batch-add", adding them all
        to the model in order.

        Args:
            expression_or_iterable (str or Iterable): The expression or sequence of expressions to append to the model
        """
        if isinstance(expression_or_iterable, string_types):
            self._append_single(expression_or_iterable)
        else:
            # Assume that it's an iterable
            self._batch_add_expressions(expression_or_iterable)

    def _append_single(self, expression):
        expr_wrapper = Expression(expression)
        self._expressions.append(expr_wrapper)
        self._modify_event()

    def _batch_add_expressions(self, list_of_expressions):
        for expr in list_of_expressions:
            expr_wrapper = Expression(expr)
            self._expressions.append(expr_wrapper)
        self._modify_event()

    def insert(self, expression, index):
        """
        Inserts an expression into the LogitModel scope at a given location. Expressions are assumed to be
        order-dependant, just in case.

        Args:
            expression (str): The expression to insert.
            index (int): The 0-based position in which to insert the expression.
        """
        expr_wrapper = Expression(expression)
        self._expressions.insert(index, expr_wrapper)
        self._modify_event()

    def remove_expression(self, index):
        """
        Removes an expression at the provided index.

        Args:
            index (int): The 0-based index at which to remove an expression
        """
        del self._expressions[index]
        self._modify_event()

    def clear(self):
        """
        Clears the LogitModel of all expressions. Any symbols already filled in the Scope will be cleared as well.
        """
        self._expressions.clear()
        self._modify_event()

    def _modify_event(self):
        self._modified = True
        self._model_ref.scope.clear()

    def get_symbols(self):
        if self._modified or self._cached_types is None or self._cached_literals is None:
            symbol_types = {}
            dict_literals = {}
            for alias, list_of_usages in iteritems(self._all_symbols()):
                symbol_type = self._check_symbol(alias, list_of_usages)

                if symbol_type is DictLiteral:
                    dict_literals[alias] = list_of_usages[0]  # Dict literals only have one use case
                else:
                    symbol_types[alias] = symbol_type
            self._modified = False
            self._cached_types = symbol_types
            self._cached_literals = dict_literals
            return symbol_types, dict_literals
        else:
            return self._cached_types, self._cached_literals

    @staticmethod
    def _check_symbol(alias, list_of_usages):
        """
        Checks a symbol's usages to ensure that they are consistently used; then return the inferred type of that symbol
        """
        inferred_type = None
        for usage in list_of_usages:
            usage_type = usage.__class__

            if inferred_type is None:
                inferred_type = usage_type
            else:
                usage_is_attributed = usage_type == AttributedUsage
                inferred_is_attributed = inferred_type == AttributedUsage
                usage_is_linked = usage_type == LinkedFrameUsage
                inferred_is_linked = inferred_type == LinkedFrameUsage

                if usage_is_linked and inferred_is_attributed:
                    inferred_type = LinkedFrameUsage
                elif usage_is_attributed and inferred_is_linked:
                    pass
                elif usage_type != inferred_type:
                    raise InconsistentUsageError(
                        "Symbol '%s' is used inconsistently" % alias
                    )
        if inferred_type is None:
            raise RuntimeError("Inferred type should never be None")

        return inferred_type

    def _all_symbols(self):
        all_symbols = {}
        for expression in self._expressions:
            for alias, usages in expression.symbols():
                if isinstance(usages, DictLiteral):
                    usages = [usages]
                if alias not in all_symbols:
                    all_symbols[alias] = list(usages)  # Make a copy
                else:
                    all_symbols[alias] += usages
        return all_symbols
