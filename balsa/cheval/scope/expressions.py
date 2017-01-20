

class ExpressionContainer(object):

    def append_expression(self, expression):
        raise NotImplementedError

    def insert_expression(self, expression, index):
        raise NotImplementedError

    def remove_expresion(self, expression=None, index=None):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError()


