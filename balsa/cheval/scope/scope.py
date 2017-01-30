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


class ScopeOrientationError(IndexError):
    pass


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
        self._records = pd.Index(index)

    def empty_symbols(self) -> Iterable[str]:
        """
        Iterate through the set of empty (not yet filled) symbols.

        Returns:
            Iterable[str]: Symbols that are empty (have no set data just yet)

        """
        raise NotImplementedError()

    def fill_symbol(self, symbol_name: str, data: Any, orientation: int=None, strict=True):
        """
        Associate an empty symbol with actual data, usually ararys, DataFrames, and/or Series. Symbol usages are
        collected from the list of model expressions already-loaded, so the type of `data` must conform to the rules of
        usage.

        Args:
            symbol_name (str): The name of the symbol being filled.
            data: Numerical data to be associated with the filled symbol. Supported types include scalar numbers, 1-D
                arrays, 2-D arrays, Series, DataFrames, and Dictionaries of DataFrames / Panels. Not all data types are
                supported by all use cases, refer to usage rules for further details.
            orientation (int): Specifies which dimension of the utility table to which the FIRST dimension (axis)
                of the data is oriented: 0: oriented to the records, 1: oriented to the alternatives. This is useful in
                cases where the records index and alternatives index are the same (the labels are therefore ambiguous),
                or the same length (for adding unlabelled data).
            strict (bool): In the case that Cheval is unable to recognize or support the type of `data`, turning off
                `strict` will permit the data to be filled at this time, and checked by the NumExpr engine later. In
                most cases, this should probably be left to True.

        Raises:
            KeyError: If the symbol_name is not found used in any Expression.
            TypeError: If the type of `data` was not understood or not supported by the scoping rules.
            ScopeOrientationError: If the data do not conform to the rules for scoping.

        """
        self._initialize()

        symbol_usage = self._empty_symbols[symbol_name]

        if isinstance(symbol_usage, LinkedFrameUsage):
            symbol_meta = self._fill_linked(data, orientation)
        elif isinstance(symbol_usage, AttributedUsage):
            symbol_meta = self._fill_attributed(data, orientation)
        elif isinstance(symbol_usage, DictLiteral):
            symbol_meta = self._fill_simple(symbol_usage.series, orientation=1)
        elif isinstance(symbol_usage, SimpleUsage):
            symbol_meta = self._fill_simple(data, orientation, strict)
        else:
            raise NotImplementedError("Usage type '%s' not understood" % type(symbol_usage))

        self._filled_symbols[symbol_name] = symbol_meta

    def _fill_simple(self, data, orientation: int=None, strict=True):
        """"""

        '''
        This is a bit ugly and should probably be broken out into separate functions for different types
        '''

        if isinstance(data, np.ndarray):
            # Unlabelled 1- or 2-D array
            self._check_records()

            if data.ndim == 1:
                # 1-D array
                n_rows = len(data)
                n_rec = len(self._records)
                n_alts = len(self._alternatives)
                if n_rec == n_alts:
                    # Orientation matters
                    if n_rows != n_rec:
                        raise ScopeOrientationError("1D data array shape incompatible with records or alternatives")
                    if orientation is None:
                        raise ScopeOrientationError("Orientation must be provided for arrays when length of records "
                                                    "matches length of alternatives")
                    return Array1DSymbol(data, orientation)
                else:
                    # Infer orientation from length of array
                    if n_rows == n_rec:
                        # Oriented to the records
                        return Array1DSymbol(data, 0)
                    elif n_rows == n_alts:
                        return Array1DSymbol(data, 1)
                    else:
                        raise ScopeOrientationError("1D arrays must be as long as either the records or alternatives")
            elif data.ndim == 2:
                # 2-D array
                n_rows, n_cols = data.shape
                n_rec, n_alts = len(self._records), len(self._alternatives)

                if n_rec == n_alts:
                    # Orientation matters
                    if n_rows != n_rec or n_cols != n_alts:
                        raise ScopeOrientationError("2D data array shape incompatible with records and alternatives")
                    if orientation is None:
                        raise ScopeOrientationError("Orientation must be provided for arrays when length of records "
                                                    "matches length of alternatives")
                    return Array2DSymbol(data, orientation)
                elif n_rows == n_rec and n_cols == n_alts:
                    return Array2DSymbol(data, 0)
                elif n_rows == n_alts and n_cols == n_rec:
                    return Array2DSymbol(data, 1)
                else:
                    raise ScopeOrientationError("2D array shapes must align with both the records and alternatives")
            else:
                raise ScopeOrientationError("Numpy arrays are permitted, but only with 1 or 2 dimensions (found more)")
        elif isinstance(data, pd.DataFrame):
            # Labelled 2-D array
            self._check_records()
            # For a simple symbols, this is only permitted if BOTH axes align with the utilities table
            if self._records.equals(self._alternatives):
                # Orientation is ambiguous
                self._check_orientation(orientation)
                if not data.index.equals(self._records) or not data.columns.equals(self._records):
                    raise ScopeOrientationError("Simple DataFrames must align with both the records and alternatives")
                return Array2DSymbol(data.values, orientation)
            elif data.index.equals(self._records) and data.columns.equals(self._alternatives):
                return Array2DSymbol(data.values, 0)
            elif data.index.equals(self._alternatives) and data.columns.equals(self._records):
                return Array2DSymbol(data.values, 1)
            else:
                raise ScopeOrientationError("Simple DataFrames must align with both the records and alternatives")
        elif isinstance(data, pd.Series):
            # Labelled 1-D array
            self._check_records()

            if self._records.equals(self._alternatives):
                # Orientation is ambiguous
                self._check_orientation(orientation)
                if not data.index.equals(self._records):
                    raise ScopeOrientationError("Series must align with either the records or alternatives")

                return Array1DSymbol(data.values, orientation)
            elif data.index.equals(self._records):
                return Array1DSymbol(data.values, 0)
            elif data.index.equals(self._alternatives):
                return Array1DSymbol(data.values, 1)
            else:
                raise ScopeOrientationError("Series must align with either the records or the alternatives")
        elif strict:
            # Rigourous type-checking for scalars if strict is required
            if not isinstance(data, (int, float, np.int_, np.float_)):
                raise TypeError("Unsupported simple symbol data type: %s" % type(data))
            return ScalarSymbol(data)
        else:
            # Permissive; let NumExpr figure out if the data is valid
            return ScalarSymbol(data)

    @staticmethod
    def _check_orientation(orientation):
        if orientation is None:
            raise ScopeOrientationError("Orientation must be provided when the record index exactly equals the"
                                        " alternatives index.")

    def _check_records(self):
        assert self._records is not None, "This operation is not allowed if the records have not been set."

    def _fill_attributed(self, data, orientation: int=None):
        self._check_records()

        if isinstance(data, pd.DataFrame):
            if self._records.equals(self._alternatives):
                self._check_orientation(orientation)
                if not data.index.equals(self._records):
                    raise ScopeOrientationError("Filling attributed usage with a DataFrame is permitted, but the index "
                                                "must align with either the records or the alternatives")
                return FrameSymbol(data, orientation)
            elif data.index.equals(self._records):
                return FrameSymbol(data, 0)
            elif data.index.equals(self._alternatives):
                return FrameSymbol(data, 1)
            else:
                raise ScopeOrientationError("Filling attributed usage with a DataFrame is permitted, but the index "
                                            "must align with either the records or the alternatives")

        elif isinstance(data, (dict, pd.Panel)):
            if isinstance(data, dict): data = pd.Panel(dict)
            raise NotImplementedError()
        else:
            raise TypeError("Only DataFrames, dictionaries of DataFrames, or Panels can fill attributed symbols")

    def _fill_linked(self, data, orientation: int):
        assert isinstance(data, LinkedDataFrame), "Only LinkedDataFrames can fill linked symbols."
        self._check_records()

        if self._records.equals(self._alternatives):
            self._check_orientation(orientation)

            if not data.index.equals(self._records):
                raise ScopeOrientationError("Filling linked usage with a LinkedDataFrame is permitted, but the index "
                                            "must align with either the records or alternatives")

            return LinkedFrameSymbol(data, orientation)
        elif data.index.equals(self._records):
            return LinkedFrameSymbol(data, 0)
        elif data.index.equals(self._alternatives):
            return LinkedFrameSymbol(data, 1)
        else:
            raise ScopeOrientationError("Filling linked usage with a LinkedDataFrame is permitted, but the index must "
                                        "align with either the records or alternatives")

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

    def get_value(self, usage: SimpleUsage):
        return self._data


class FrameSymbol(AbstractSymbol):

    def __init__(self, frame: pd.DataFrame, orientation: int):
        self._frame = frame
        self._orientation = orientation

    def get_value(self, usage: AttributedUsage):
        raise NotImplementedError()


class LinkedFrameSymbol(AbstractSymbol):
    def __init__(self, frame: LinkedDataFrame, orientation: int):
        self._frame = frame
        self._orientation = orientation

    def get_value(self, usage: LinkedFrameUsage):
        raise NotImplementedError()


class PanelSymbol(AbstractSymbol):

    def __init__(self):
        raise NotImplementedError()

    def get_value(self, usage: AttributedUsage):
        raise NotImplementedError()
