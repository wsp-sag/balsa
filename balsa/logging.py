from __future__ import division, absolute_import, print_function, unicode_literals

import logging
import sys
from contextlib import contextmanager
import traceback as tb

_DEFAULT_FMT = "%(asctime)s %(levelname)s %(name)s -> %(message)s"
_COLOUR_CODES = {'red': 31, 'green': 32, 'gold': 33, 'yellow': 33, 'blue': 34}

# region Custom levels
TIP_LEVEL = 21  # Important log info
logging.addLevelName(TIP_LEVEL, 'TIP')

REPORT_LEVEL = 19  # Model results
logging.addLevelName(REPORT_LEVEL, 'REPORT')
# endregion

# region Classes


class _SetFilter(object):

    def __init__(self):
        self._excluded = set()

    def filter(self, record):
        return record.levelno not in self._excluded

    def exclude(self, level):
        self._excluded.add(level)

    def include(self, level):
        self._excluded.remove(level)


class _RangeFilter(object):

    def __init__(self, low, high):
        self._low = int(low)
        self._high = int(high)

    def filter(self, record):
        return self. _low <= record.levelno <= self._high


class _SwitchFormatter(logging.Formatter):

    def __init__(self, default_format):
        super(_SwitchFormatter, self).__init__()
        self._default = logging.Formatter(default_format)
        self._formats = {}

    def format(self, record):
        level = record.levelno
        if level in self._formats:
            return self._formats[level].format(record)
        return self._default.format(record)

    def add(self, level, fmt):
        self._formats[level] = logging.Formatter(fmt)

    def change_default(self, fmt):
        self._default = logging.Formatter(fmt)

    def clear(self):
        self._formats.clear()


class ModelLogger(logging.Logger):
    """
    Extends the standard Python Logger object, adding additional logging statements such as ``.report()``.
    """

    def report(self, msg, *args, **kwargs):
        """Report useful model statistics or results to the user. Distinct from ``.info()`` which provides status
        information. Printed in green when colours are available."""
        self.log(REPORT_LEVEL, msg, *args, **kwargs)

    def tip(self, msg, *args, **kwargs):
        """Provide a more significant status statement (e.g. new section of the model). Similar to ``.info()``, but
        more emphasized. Printed in blue when colours are available."""
        self.log(TIP_LEVEL, msg, *args, **kwargs)

    def pipe(self, msg, levelno=0, levelname=None, asctime=None, name=None):
        """Turn a string message into a logging statement. Designed to work with stdout when working with
        sub-processes."""
        d = {"msg": msg}
        if levelno is not None: d['levelno'] = levelno
        if levelname is not None: d['levelname'] = levelname
        if asctime is not None: d['asctime'] = asctime
        if name is not None: d['name'] = name
        else: d['name'] = self.name

        rec = logging.makeLogRecord(d)
        self.handle(rec)

# endregion

# region Global variables & functions


def set_console_format(fmt: str, level: int=None, colour: str=None):
    if colour is not None:
        code = _COLOUR_CODES[colour.lower()]
        fmt = "\x1b[{code}m{fmt}\x1b[0m".format(code=code, fmt=fmt)

    if level is not None:
        _CONSOLE_FORMATTER.add(level, fmt)
    else:
        _CONSOLE_FORMATTER.clear()
        _CONSOLE_FORMATTER.change_default(fmt)


def exclude_console_level(level):
    _CONSOLE_FILTER.exclude(level)


def include_console_level(level):
    _CONSOLE_FILTER.include(level)


_CONSOLE_FILTER = _SetFilter()
_CONSOLE_FORMATTER = _SwitchFormatter(_DEFAULT_FMT)

_STDOUT_HANDLER = logging.StreamHandler(sys.stdout)
_STDOUT_HANDLER.addFilter(_CONSOLE_FILTER)
_STDOUT_HANDLER.addFilter(_RangeFilter(0, logging.ERROR - 1))
_STDOUT_HANDLER.setFormatter(_CONSOLE_FORMATTER)

_STDERR_HANDLER = logging.StreamHandler(sys.stderr)
_STDERR_HANDLER.addFilter(_CONSOLE_FILTER)
_STDERR_HANDLER.addFilter(_RangeFilter(logging.ERROR, 100))
_STDERR_HANDLER.setFormatter(_CONSOLE_FORMATTER)

set_console_format(_DEFAULT_FMT, level=TIP_LEVEL, colour='blue')
set_console_format(_DEFAULT_FMT, level=REPORT_LEVEL, colour='green')
set_console_format(_DEFAULT_FMT, level=logging.WARNING, colour='red')

# endregion

# region Loggers


def init_root(root_name: str) -> ModelLogger:
    """
    Call this function **at the start of a program** to setup Balsa's logging features, including colours.

    Args:
        root_name: Loggers use dotted namespaces to determine inheritance, which also gets passed into the logging
            statements. So use a descriptive name for your program, with no spaces in it. Subsequent loggers should
            include this name, then a dot, and then the model-specific name.

    Returns:
        ModelLogger: The root ModelLogger object.

    """
    logging.setLoggerClass(ModelLogger)
    root = logging.getLogger(root_name)
    root.propagate = True
    root.setLevel(1)  # Log everything

    root.handlers.clear()  # Remove any existing handlers
    root.addHandler(_STDOUT_HANDLER)
    root.addHandler(_STDERR_HANDLER)

    return root


def get_model_logger(name: str) -> ModelLogger:
    logging.setLoggerClass(ModelLogger)
    return logging.getLogger(name)


@contextmanager
def log_to_file(file_name: str, name, *, append=False):
    root = logging.getLogger(name)

    write_mode = 'a' if append else 'w'
    handler = logging.FileHandler(file_name, mode=write_mode)
    handler.setFormatter(logging.Formatter(_DEFAULT_FMT))
    handler.addFilter(_RangeFilter(0, 100))

    root.addHandler(handler)
    try:
        yield
    except:
        with open(file_name, mode='a') as writer:
            writer.write("\n" + "-" * 100 + "\n\n")
            writer.write(tb.format_exc())
        raise
    finally:
        root.removeHandler(handler)

# endregion
