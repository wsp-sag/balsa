import unittest

from numpy.random import uniform, choice
import numpy as np
import pandas as pd

from balsa.cheval import ChoiceModel, ScopeOrientationError
from balsa.cheval.scope.scope import (ScalarSymbol, Array1DSymbol, Array2DSymbol, FrameSymbol, LinkedFrameSymbol,
                                      PanelSymbol)
from balsa.cheval.scope import SimpleUsage, AttributedUsage, LinkedFrameUsage
from balsa.cheval.ldf import LinkedDataFrame


class TestScope(unittest.TestCase):

    def setUp(self):
        self.__record_index = pd.Index(range(10))
        self.__alts_index = pd.Index(list('abcdefg'))

        model = ChoiceModel()
        for char in 'abcdefg':
            model.tree.add_node(char)

        self.__model = model

    def __prep_for_testing(self, expression):
        self.__model.expressions.clear()
        self.__model.expressions.append_expression(expression)
        self.__model.scope.set_record_index(self.__record_index)

    def test_scalar_symbol(self):
        self.__prep_for_testing('a * 2')
        model = self.__model

        model.scope.fill_symbol('a', 0.5)
        assert 'a' in model.scope._filled_symbols
        symbol = model.scope._filled_symbols['a']
        assert isinstance(symbol, ScalarSymbol)
        assert symbol.get_value(None) == 0.5

    def test_array_1d_symbol(self):
        self.__prep_for_testing('a + b + c + d')
        model = self.__model

        data1 = uniform(low=-1.0, high=1.0, size=10)
        model.scope.fill_symbol('a', data1)
        symbol_a = model.scope._filled_symbols['a']
        assert isinstance(symbol_a, Array1DSymbol)
        assert symbol_a._orientation == 0
        value = symbol_a.get_value(None)
        assert value.shape == (10, 1)

        data2 = uniform(low=-1.0, high=1.0, size=7)
        model.scope.fill_symbol('b', data2)
        symbol_b = model.scope._filled_symbols['b']
        assert isinstance(symbol_b, Array1DSymbol)
        assert symbol_b._orientation == 1
        value = symbol_b.get_value(None)
        assert value.shape == (1, 7)

        model.scope.fill_symbol('c', pd.Series(data1, index=self.__record_index))
        symbol_c = model.scope._filled_symbols['c']
        assert isinstance(symbol_c, Array1DSymbol)
        assert symbol_c._orientation == 0
        value = symbol_c.get_value(None)
        assert value.shape == (10, 1)

        model.scope.fill_symbol('d', pd.Series(data2, self.__alts_index))
        symbol_d = model.scope._filled_symbols['d']
        assert isinstance(symbol_d, Array1DSymbol)
        assert symbol_d._orientation == 1
        value = symbol_d.get_value(None)
        assert value.shape == (1, 7)

    def test_array_2d_symbol(self):
        self.__prep_for_testing('d + e + f + g')
        model = self.__model

        # 3. Test Array2DSymbol from unlabelled arrays
        data3 = uniform(size=(10, 7))
        model.scope.fill_symbol('d', data3)
        symbol_d = model.scope._filled_symbols['d']
        assert isinstance(symbol_d, Array2DSymbol)
        assert symbol_d._data.shape == (10, 7)
        value = symbol_d.get_value(SimpleUsage())
        assert value.shape == (10, 7)

        data4 = uniform(size=(7, 10))
        model.scope.fill_symbol('e', data4)
        symbol_e = model.scope._filled_symbols['e']
        assert isinstance(symbol_e, Array2DSymbol)
        assert symbol_e._data.shape == (10, 7)
        value = symbol_e.get_value(SimpleUsage())
        assert value.shape == (10, 7)

        # 4. Test Array2DSymbol from labelled arrays
        data5 = pd.DataFrame(uniform(size=(10, 7)), index=self.__record_index, columns=self.__alts_index)
        model.scope.fill_symbol('f', data5)
        symbol_f = model.scope._filled_symbols['f']
        assert isinstance(symbol_f, Array2DSymbol)
        assert symbol_f._data.shape == (10, 7)
        value = symbol_f.get_value(SimpleUsage())
        assert value.shape == (10, 7)

        model.scope.fill_symbol('g', data5.transpose())
        symbol_g = model.scope._filled_symbols['g']
        assert isinstance(symbol_g, Array2DSymbol)
        assert symbol_g._data.shape == (10, 7)
        value = symbol_g.get_value(SimpleUsage())
        assert value.shape == (10, 7)

    def test_frame_symbol(self):
        self.__prep_for_testing('a.b + c.d')
        model = self.__model

        # 1. Fill with record-oriented DataFrame
        data1 = pd.DataFrame({'b': uniform(size=10), 'colour': choice(['red', 'blue'], size=10, replace=True)},
                             index=self.__record_index)
        data1['colour'] = data1['colour'].astype('category')
        model.scope.fill_symbol('a', data1)
        symbol_a = model.scope._filled_symbols['a']
        assert isinstance(symbol_a, FrameSymbol)
        assert symbol_a._orientation == 0
        numeric_value = symbol_a.get_value(AttributedUsage('', 'b'))
        assert numeric_value.shape == (10, 1)
        cat_value = symbol_a.get_value(AttributedUsage('', 'colour'))
        assert cat_value.dtype == np.dtype('a4')

        # 2. Fill with alternatives-oriented DataFrame
        data2 = pd.DataFrame({'d': uniform(size=7)}, index=self.__alts_index)
        model.scope.fill_symbol('c', data2)
        symbol_c = model.scope._filled_symbols['c']
        assert isinstance(symbol_c, FrameSymbol)
        assert symbol_c._orientation == 1
        value = symbol_c.get_value(AttributedUsage('', 'd'))
        assert value.shape == (1, 7)

    def test_linked_frame_symbol(self):
        self.__prep_for_testing('record.height + record.zone.area + record.hats.count(colour == "red")')
        model = self.__model

        df = LinkedDataFrame({'height': uniform(high=3, size=10),
                              'zone': choice(range(101, 104), size=10, replace=True)},
                             index=self.__record_index)
        zones = LinkedDataFrame({'area': uniform(100, 500, size=3)}, index=range(101, 104))
        hats = LinkedDataFrame({'colour': choice(['red', 'blue'], size=15, replace=True),
                                'record': [0, 0, 1, 2, 2, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10]})

        df.link_to(zones, 'zone', on_self='zone')
        df.link_to(hats, 'hats', on_other='record')

        hats.link_to(df, 'record', on_self='record')

        model.scope.fill_symbol('record', df)
        symbol = model.scope._filled_symbols['record']
        assert isinstance(symbol, LinkedFrameSymbol)
        assert symbol._orientation == 0

        for expr in model.expressions:
            assert 'record' in expr._symbols
            for usage in expr._symbols['record']:
                value = symbol.get_value(usage)
                assert value.shape == (10, 1)

    def test_panel_symbol(self):
        self.__prep_for_testing('e.f + 2')
        model = self.__model

        data3 = pd.Panel({'f': uniform(size=[10, 7])}, major_axis=self.__record_index, minor_axis=self.__alts_index)
        model.scope.fill_symbol('e', data3)
        symbol_e = model.scope._filled_symbols['e']
        assert isinstance(symbol_e, PanelSymbol)

if __name__ == '__main__':
    unittest.main()
