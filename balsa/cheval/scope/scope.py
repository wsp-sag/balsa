from ..api import LogitModel


class Scope(object):

    def __init__(self, model: LogitModel):
        self._root = model

    def get_symbol_spec(self):
        raise NotImplementedError()

    def fill_symbol(self, symbol_name, data, type=None, orientation=None):
        raise NotImplementedError()

    def clear(self):
        raise NotImplementedError()
