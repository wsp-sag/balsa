import ast
from collections import namedtuple, deque
import astor
import numpy as np
import pandas as pd
from numexpr import expressions as nee
from ..ldf import SUPPORTED_AGGREGATIONS

NUMEXPR_FUNCTIONS = set(nee.functions.keys())
MAX_ATTRIBUTE_CHAINS = 50  # To avoid infinite loop in __get_name_from_attribute()


class SimpleUsage(object):
    pass


class DictLiteral(object):
    def __init__(self, substitution, series):
        self.substitution = substitution
        self.series = series


class AttributedUsage(object):
    def __init__(self, substitution, attribute):
        self.substitution = substitution
        self.attribute = attribute


class LinkedFrameUsage(object):
    def __init__(self, substitution, stack, func, func_expr):
        self.substitution = substitution
        self.stack = stack
        self.func = func
        self.func_expr = func_expr


class UnsupportedSyntaxError(SyntaxError):
    pass


class ExpressionProcessor(ast.NodeTransformer):

    # Only nodes used in expressions are included, due to the limited parsing
    UNSUPPORTED_NODES = (
        ast.Load, ast.Store, ast.Del, ast.Starred, ast.IfExp, ast.Subscript, ast.ListComp, ast.DictComp
    )

    @staticmethod
    def parse(expression: str):
        tree = ast.parse(expression, mode='eval').body
        transformer = ExpressionProcessor()
        new_tree = transformer.visit(tree)
        new_expression = astor.to_source(new_tree)

        return new_expression, transformer.__symbols

    def __init__(self):
        self.__symbols = {}
        self.__n_dicts = 0

    def __append_symbol(self, name, usage):
        if name not in self.__symbols:
            self.__symbols[name] = [usage]
        else:
            self.__symbols[name].append(usage)

    def __generate_substitution(self, name):
        return name + (0 if name not in self.__symbols else len(self.__symbols[name]))

    def visit(self, node):
        return self.__get_visitor(node)

    def __get_visitor(self, node):
        if isinstance(node, ExpressionProcessor.UNSUPPORTED_NODES):
            raise UnsupportedSyntaxError(node.__class__.__name__)
        name = "visit_" + node.__class__.__name__.lower()
        return getattr(self, name) if hasattr(self, name) else self.generic_visit

    def visit_call(self, node: ast.Call):
        func_node = node.func

        if isinstance(func_node, ast.Name):
            # Top-level function
            return self.__visit_toplevel_func(node, func_node)
        elif isinstance(func_node, ast.Attribute):
            # Method of an object
            return self.__visit_method(node, func_node)
        else:
            return self.generic_visit(node)

    def __visit_toplevel_func(self, node: ast.Call, func_node: ast.Name):
        func_name = func_node.id
        if func_name not in NUMEXPR_FUNCTIONS:
            raise UnsupportedSyntaxError("Function '%s' not supported." % func_name)

        node.args = [self.__get_visitor(arg)(arg) for arg in node.args]
        return node

    def __visit_method(self, call_node: ast.Call, func_node: ast.Attribute):
        name, stack = self.__get_name_from_attribute(func_node)
        func_name = stack.popleft()
        if func_name not in SUPPORTED_AGGREGATIONS:
            raise UnsupportedSyntaxError("Linked Data Frame aggregation '%s' is not supported." % func_name)

        if len(call_node.keywords) > 0:
            raise UnsupportedSyntaxError("Keyword args are not supported inside Linked Data Frame aggregations")
        if call_node.starargs is not None or call_node.kwargs is not None:
            raise UnsupportedSyntaxError("Star-args or star-kwargs are not supported inside Linked Data Frame "
                                         "aggregation")
        arg_expression = astor.to_source(call_node.args)
        substitution = self.__generate_substitution(name)

        usage = LinkedFrameUsage(substitution, stack, func_name, arg_expression)
        self.__append_symbol(name, usage)

        new_node = ast.Name(substitution)
        return new_node

    def visit_name(self, node: ast.Name):
        # Register the symbol but do not change it.
        symbol_name = node.id
        self.__append_symbol(symbol_name, SimpleUsage())
        return node

    def visit_attribute(self, node: ast.Attribute):
        name, stack = self.__get_name_from_attribute(node)
        substitution = self.__generate_substitution(name)
        if len(stack) == 1:
            attribute = stack.pop()
            usage = AttributedUsage(substitution, attribute)
            self.__append_symbol(name, usage)
        else:
            usage = LinkedFrameUsage(substitution, stack, None, None)
            self.__append_symbol(name, usage)

    @staticmethod
    def __get_name_from_attribute(node: ast.Attribute):
        current_node = node
        stack = deque()
        while not isinstance(current_node, ast.Name):
            if not isinstance(current_node, ast.Attribute):
                raise UnsupportedSyntaxError()
            if len(stack) > MAX_ATTRIBUTE_CHAINS:
                raise RecursionError()
            stack.append(current_node.attr)
            current_node = current_node.value

        return current_node.id, stack

    def visit_dict(self, node: ast.Dict):
        substitution = '__dict%s' % self.__n_dicts
        self.__n_dicts += 1
        new_node = ast.Name(substitution)

        try:
            values = [np.float32(val.n) for val in node.values]
            keys = [self.__get_dict_key(key) for key in node.keys]
        except ValueError:
            raise UnsupportedSyntaxError("Dict literals are supported for numeric values only")

        s = pd.Series(values, index=keys)
        usage = DictLiteral(substitution, s)
        self.__symbols[substitution] = usage

        return new_node

    @staticmethod
    def __get_dict_key(node):
        if isinstance(node, ast.Name):
                return node.id
        if isinstance(node, ast.Str):
            return node.s
        raise UnsupportedSyntaxError("Dict key of type '%s' unsupported" % node)
