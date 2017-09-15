from typing import Union
from logging import Logger

import numpy as np
import pandas as pd

from balsa.cheval.scope import Scope, ExpressionContainer
from balsa.cheval.tree import ChoiceTree

class ChoiceModel:

    _expression_container: ExpressionContainer
    _scope_container: Scope
    _tree_container: ChoiceTree

    def __init__(self):
        pass

    @property
    def scope(self) -> Scope:
        pass

    @property
    def expressions(self) -> ExpressionContainer:
        pass

    @property
    def tree(self) -> ChoiceTree:
        pass

    def run_discrete(self, randomizer: Union[np.random.RandomState, int]=None, n_draws: int=1,
                     astype: Union[str, np.dtype, type] ='category', squeeze: bool=True, n_threads :int=1,
                     override_utilities: pd.DataFrame=None, logger: Logger=None
                     ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        pass

    def run_stochastic(self, n_threads: int=1, override_utilities: pd.DataFrame=None, logger: Logger=None
                       ) -> pd.DataFrame:
        pass

    def _check_model_is_ready(self, compute_utilities=True) -> bool:
        pass

    def _eval_probabilities_and_sample(self, utilities: np.ndarray, randomizer: np.random.RandomState, n_draws: int,
                                       n_threads: int) -> np.ndarray:
        pass

    def _convert_result(self, results: np.ndarray, astype: Union[str, np.dtype, type], squeeze: bool
                        ) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        pass

    def _eval_probabilities_only(self, utilities: np.ndarray, n_threads: int) -> np.ndarray:
        pass

    def _prep_override_utilities(self, override_utilities: pd.DataFrame) -> np.ndarray:
        pass

def sample_from_weights(weights: pd.DataFrame, randomizer: Union[np.random.RandomState, int],
                        astype: Union[str, np.dtype, type]='category', n_threads: int=1
                        ) -> pd.Series:
    pass
