import pandas as pd
import numpy as np
from typing import List, Union, Generator, Iterable, Any

class Expression:
    def __init__(self, expr: str):
        pass

class ExpressionContainer:

    _expressions: list
    _model_ref: Any

    def __init__(self, model):
        pass

    def __iter__(self) -> Generator[Expression]:
        pass

    def append(self, expression_or_iterable: Union[str, Iterable[str]]):
        pass

    def clear(self):
        pass

    def _append_single(self, expression):
        pass

    def _batch_add_expressions(self, list_of_expressions):
        pass

    def _modify_event(self):
        pass
