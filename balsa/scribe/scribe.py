from __future__ import division, absolute_import, print_function, unicode_literals

import logging
import sys
from contextlib import contextmanager
import traceback as tb

REPORT_LEVEL = 15
logging.addLevelName(REPORT_LEVEL, 'REPORT')


class ModelLogger(logging.Logger):

    def report(self, msg, *args, **kwargs):
        """Report interim model results"""
        self.log(REPORT_LEVEL, msg, *args, **kwargs)


class RangeFilter(object):

    def __init__(self, low, high):
        self._low = int(low)
        self._high = int(high)

    def filter(self, record):
        return self. _low <= record.levelno <= self._high


logging.setLoggerClass(ModelLogger)
_CURRENT_ROOT = ''
_MODEL_FORMAT = "%(asctime)s %(levelname)s %(name)s -> %(message)s"


def get_root_logger(root_name):
    global _CURRENT_ROOT

    root = logging.getLogger(root_name)
    root.setLevel(1)
    root.propagate = True

    formatter = logging.Formatter(_MODEL_FORMAT)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(1)
    ch.setFormatter(formatter)
    ch.addFilter(RangeFilter(logging.DEBUG, logging.INFO))
    root.addHandler(ch)

    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(1)
    ch.setFormatter(formatter)
    ch.addFilter(RangeFilter(logging.WARNING, logging.CRITICAL))
    root.addHandler(ch)

    _CURRENT_ROOT = root_name

    return root


def get_model_logger(name):
    return logging.getLogger(_CURRENT_ROOT + '.' + name)


def remove_jupyter_handler():
    """
    Removes the default Jupyter handler.

    Jupyter Notebooks add a handler which logs everything to STDERR. This results in a doubling of logging output from
    model code. Calling this function fixes this problem so that only one logging line is printed.
    """

    blank_logger = logging.getLogger()
    if len(blank_logger.handlers) > 0:
        blank_logger.removeHandler(blank_logger.handlers[0])


@contextmanager
def log_to_file(log_file, logger=None):
    """
    Context manager for adding a file handler to the logging system. If logger is not provided, it will default to the
    root logger.

    Args:
        log_file: Path to the log file.
        logger (Logger): An optional logger to add the file handler to.
    """

    if logger is None:
        logger = logging.getLogger(_CURRENT_ROOT)

    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(logging.Formatter(_MODEL_FORMAT))
    fh.addFilter(RangeFilter(0, 100))

    logger.addHandler(fh)

    try:
        yield
    except:
        with open(log_file, mode='a') as writer:
            writer.write("\n" + "-" * 100 + "\n\n")
            writer.write(tb.format_exc())
        raise
    finally:
        logger.removeHandler(fh)
        fh.close()
