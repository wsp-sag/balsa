from typing import Union, Dict
from .parsing import ExpressionProcessor, SimpleSymbol, DictLiteral, AttributedSymbol, LinkedFrameSymbol
from ..api import LogitModel
from six import iteritems


class InconsistentUsageError(RuntimeError):
    pass


class Expression(object):

    def __init__(self, expr: str):
        self._raw_expr = expr
        parsed, symbols = ExpressionProcessor.parse(expr)
        self._parsed_expr = parsed
        self._symbols = symbols

    def symbols(self):
        yield from iteritems(self._symbols)


class ExpressionContainer(object):

    def __init__(self, model: LogitModel):
        self._expressions = []
        self._model_ref = model
        self._modified = True
        self._cached_types = None

    def append_expression(self, expression: str):
        expr_wrapper = Expression(expression)
        self._expressions.append(expr_wrapper)
        self._modify_event()

    def insert_expression(self, expression: str, index: int):
        expr_wrapper = Expression(expression)
        self._expressions.insert(index, expr_wrapper)
        self._modify_event()

    def remove_expression(self, index: int):
        del self._expressions[index]
        self._modify_event()

    def batch_add_expressions(self, list_of_expressions):
        for expr in list_of_expressions:
            expr_wrapper = Expression(expr)
            self._expressions.append(expr_wrapper)
        self._modify_event()

    def clear(self):
        self._expressions.clear()
        self._modify_event()

    def _modify_event(self):
        self._modified = True
        self._model_ref.scope.clear()

    def get_symbols(self) -> Dict[str, Union[SimpleSymbol, DictLiteral, AttributedSymbol, LinkedFrameSymbol]]:
        if self._modified or self._cached_types is None:
            symbol_types = {}
            for alias, list_of_usages in self._all_symbols():
                symbol_type = self._check_symbol(alias, list_of_usages)
                symbol_types[alias] = symbol_type
            self._modified = False
            self._cached_types = symbol_types
            return symbol_types
        else:
            return self._cached_types

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
                usage_is_attributed = usage_type == AttributedSymbol
                inferred_is_attributed = inferred_type == AttributedSymbol
                usage_is_linked = usage_type == LinkedFrameSymbol
                inferred_is_linked = inferred_type == LinkedFrameSymbol

                if usage_is_linked and inferred_is_attributed:
                    inferred_type = LinkedFrameSymbol
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
                if alias not in all_symbols:
                    all_symbols[alias] = list(usages)  # Make a copy
                else:
                    all_symbols[alias] += usages
        return all_symbols
