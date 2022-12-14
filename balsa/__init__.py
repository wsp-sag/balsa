from ._version import get_versions
from .routines import *

__version__ = get_versions()['version']
del get_versions
