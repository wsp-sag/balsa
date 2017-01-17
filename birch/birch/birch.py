import logging
import sys

try:
    import inro.modeller as m
except ImportError:
    m = None

REPORT_LEVEL = 15
logging.addLevelName(REPORT_LEVEL, 'REPORT')


PROXY_INFO_LEVEL = 14
PROXY_WARN_LEVEL = 31
PROXY_ERR_LEVEL = 41


class ModelLogger(logging.Logger):

    def report(self, message, *args, **kwargs):
        self.log(REPORT_LEVEL, message, *args, **kwargs)

    def proxy_info(self, message, *args, **kwargs):
        self.log(PROXY_INFO_LEVEL, message, *args, **kwargs)

    def proxy_warn(self, message, *args, **kwargs):
        self.log(PROXY_WARN_LEVEL, message, *args, **kwargs)

    def proxy_error(self, message, *args, **kwargs):
        self.log(PROXY_ERR_LEVEL, message, *args, **kwargs)


class RangeFilter(object):

    def __init__(self, low, high):
        self._low = int(low)
        self._high = int(high)

    def filter(self, record):
        return self. _low <= record.levelno <= self._high


class SingleFilter(object):
    def __init__(self, level):
        self._level = level

    def filter(self, record):
        return record.levelno == self._level


class DispatchingFormatter(logging.Formatter):

    def __init__(self, default):
        super(DispatchingFormatter, self).__init__()
        self._default_formatter = logging.Formatter(default)
        self._filters = []
        self._formatters = []

    def add_formatter(self, filter_, format_str):
        self._filters.append(filter_)
        self._formatters.append(logging.Formatter(format_str))

    def format(self, record):
        for filter_, formatter in zip(self._filters, self._formatters):
            if filter_.filter(record):
                return formatter.format(record)
        return self._default_formatter.format(record)


logging.setLoggerClass(ModelLogger)
get_model_logger = logging.getLogger

MODEL_FORMATTER = "%(asctime)s %(levelname)s %(name)s -> %(message)s"
PROXY_FORMATTER = "%(message)s"


def get_root_logger(name):
    root_logger = get_model_logger(name)
    root_logger.setLevel(1)
    root_logger.propagate = True

    all_level_filter = RangeFilter(logging.DEBUG, logging.INFO)

    formatter = DispatchingFormatter(default=MODEL_FORMATTER)
    formatter.add_formatter(SingleFilter(PROXY_INFO_LEVEL), PROXY_FORMATTER)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(1)
    handler.setFormatter(formatter)
    handler.addFilter(all_level_filter)
    root_logger.addHandler(handler)

    formatter = DispatchingFormatter(default=MODEL_FORMATTER)
    formatter.add_formatter(SingleFilter(PROXY_ERR_LEVEL), PROXY_FORMATTER)
    formatter.add_formatter(SingleFilter(PROXY_WARN_LEVEL), PROXY_FORMATTER)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(1)
    handler.setFormatter(formatter)
    handler.addFilter(all_level_filter)
    root_logger.addHandler(handler)

    return root_logger


def remove_jupyter_handler():
    """
    Removes the default Jupyter handler.

    Jupyter Notebooks add a handler which logs everything to STDERR. This results in a doubling of logging output from
    model code. Calling this function fixes this problem so that only one logging line is printed.
    """
    blank_logger = get_model_logger()
    if len(blank_logger.handlers) > 0:
        blank_logger.removeHandler(blank_logger.handlers[0])