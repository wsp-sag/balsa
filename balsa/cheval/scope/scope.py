from enum import Enum
import abc
from typing import Any, Iterable, Union

import pandas as pd
import numpy as np

from ..api import LogitModel
from .expressions import SimpleUsage, DictLiteral, AttributedUsage, LinkedFrameUsage
from ..ldf import LinkedDataFrame


class Orientation(Enum):
    RECORDS = 0
    ALTERNATIVES = 1


class Scope(object):

    def __init__(self, model: LogitModel):
        self._root = model
        self._empty_symbols = None
        self._filled_symbols = None

        self._records = None
        self._alternatives = None

    def set_record_index(self, index: Iterable):
        """
        Manually set the index of records being processed.

        If some symbols are already filled, calling this method CLEARS their data, resetting them back to 'empty'.

        Args:
            index (List or Index): Any sequence that Pandas accepts when constructing an Index.

        """
        self._initialize()
        raise NotImplementedError()

    def empty_symbols(self) -> Iterable[str]:
        """
        Iterate through the set of empty (not yet filled) symbols.

        Returns:
            Iterable[str]: Symbols that are empty (have no set data just yet)

        """
        raise NotImplementedError()

    def fill_symbol(self, symbol_name: str, data: Any, orientation: Orientation=None, strict=True):
        """
        Associate an empty symbol with actual data, usually ararys, DataFrames, and/or Series. Symbol usages are
        collected from the list of model expressions already-loaded, so the type of `data` must conform to the rules of
        usage.

        Args:
            symbol_name (str): The name of the symbol being filled.
            data: Numerical data to be associated with the filled symbol. Supported types include scalar numbers, 1-D
                arrays, 2-D arrays, Series, DataFrames, and Dictionaries of DataFrames / Panels. Not all data types are
                supported by all use cases, refer to usage rules for further details.
            orientation (Orientation): Specifies which dimension of the utility table to which the FIRST dimension (axis)
                of the data is oriented. This is useful in cases where the records index and alternatives index are the
                same (the labels are therefore ambiguous), or to force the scope to use the first dimension of labelled
                data as the records index.
            strict (bool): In the case that Cheval is unable to recognize or support the type of `data`, turning off
                `strict` will permit the data to be filled at this time, and checked by the NumExpr engine later. In
                most cases, this should probably be left to True.

        Raises:
            KeyError: If the symbol_name is not found used in any Expression.
            TypeError: If `strict` is True, and the symbol being filled is a simple substitution, AND the type of `data`
                is not a scalar.

        """
        self._initialize()

        symbol_type = self._empty_symbols[symbol_name]

    def _fill_simple(self, symbol_meta: SimpleUsage, data, orientation: Orientation, strict=True):
        if isinstance(data, np.ndarray):
            # Unlabelled 1- or 2-D array
            self._check_records()
            raise NotImplementedError()
        elif isinstance(data, pd.DataFrame):
            # Labelled 2-D array
            # For a simple symbols, this is only permitted if BOTH axes align with the utilities table
            raise NotImplementedError()
        elif isinstance(data, pd.Series):
            # Labelled 1-D array
            raise NotImplementedError()
        elif strict:
            # Rigourous type-checking for scalars if strict is required
            if not isinstance(data, (int, float, np.int_, np.float_)):
                raise TypeError("Unsupported simple symbol data type: %s" % type(data))
            raise NotImplementedError()
        else:
            # Permissive; let NumExpr figure out if the data is valid
            raise NotImplementedError()

    def _check_records(self):
        assert self._records is not None, "This operation is not allowed if the records have not been set."

    def _fill_attributed(self, symbol_meta, data, orientation: Orientation):
        assert isinstance(data, (pd.DataFrame, pd.Panel, dict)), "Only DataFrames, Panel, and Dictionaries can fill " \
                                                                 "attributed symbols"

        raise NotImplementedError()

    def _fill_linked(self, symbol_meta, data, orientation: Orientation):
        assert isinstance(data, LinkedDataFrame), "Only LinkedDataFrames can fill linked symbols."
        raise NotImplementedError()

    def clear(self):
        self._empty_symbols = None
        self._filled_symbols = None
        self._records = None
        self._alternatives = None

    def _initialize(self):
        if self._empty_symbols is None or self._filled_symbols is None:
            self._empty_symbols = self._root.expressions.get_symbols()
            self._filled_symbols = {}
            self._alternatives = self._root.node_index


class AbstractSymbol(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_value(self, usage):
        pass


class ScalarSymbol(AbstractSymbol):

    def __init__(self, value):
        self._value = value

    def get_value(self, usage):
        return self._value


class Array1DSymbol(AbstractSymbol):

    def __init__(self, array: np.ndarray, orientation: int):
        self._data = array
        self._orientation = orientation

    def get_value(self, usage):
        length = len(self._data)
        new_shape = [1, 1]
        new_shape[self._orientation] = length

        view = self._data[...]  # Make a shallow copy
        view.shape = new_shape
        return view


class Array2DSymbol(AbstractSymbol):

    def __init__(self, array: np.ndarray, orientation: int):
        self._data = np.transpose(array) if orientation == 1 else array[...]

    def get_value(self, usage):
        return self._data


class FrameSymbol(AbstractSymbol):

    def __init__(self, frame: pd.DataFrame, orientation: int):
        self._frame = frame
        self._orientation = orientation

    def get_value(self, usage: Union[AttributedUsage, LinkedFrameUsage]):
        raise NotImplementedError()


class PanelSymbol(AbstractSymbol):

    def __init__(self):
        raise NotImplementedError()

    def get_value(self, usage):
        raise NotImplementedError()
