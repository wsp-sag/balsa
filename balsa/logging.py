from __future__ import division, absolute_import, print_function, unicode_literals

import logging
import sys
from contextlib import contextmanager
import traceback as tb
import six
from json import dumps as json_to_str
from enum import Enum

# region Constants

_FMT_STRING = "%(asctime)s %(levelname)s %(name)s {arrow} %(message)s"
_UNC_ARROW = "➔"
_ASCII_ARROW = "->"
_TIP_LEVEL = 21  # Important log info
_REPORT_LEVEL = 19  # Model results
_FANCY_FORMAT = "FANCY"
_BASIC_FORMAT = "BASIC"
_JSON_FORMAT = "JSON"


class LogFormats(Enum):

    FANCY = _FANCY_FORMAT
    BASIC = _BASIC_FORMAT
    JSON = _JSON_FORMAT


# endregion

# region Classes


class _RangeFilter(object):

    def __init__(self, low, high):
        self._low = int(low)
        self._high = int(high)

    def filter(self, record):
        return self. _low <= record.levelno <= self._high


class _SwitchFormatter(logging.Formatter):

    def __init__(self, default_format, level_formats):
        super(_SwitchFormatter, self).__init__()

        def make_formatter(item):
            return logging.Formatter(item) if isinstance(item, str) else item

        self._default = make_formatter(default_format)
        self._formats = {lvl: make_formatter(f) for lvl, f in six.iteritems(level_formats)}

    def format(self, record):
        level = record.levelno
        if level in self._formats:
            return self._formats[level].format(record)
        return self._default.format(record)


class _JsonFormatter(logging.Formatter):

    def format(self, record):
        keys = ["levelname", "name", "msg", "created"]
        to_json = {key: getattr(record, key) for key in keys}

        to_json['asctime'] = self.formatTime(record)

        return json_to_str(to_json)


class ModelLogger(logging.Logger):
    """
    Extends the standard Python Logger object, adding additional logging statements such as ``.report()``.
    """

    def report(self, msg, *args, **kwargs):
        """Report useful model statistics or results to the user. Distinct from ``.info()`` which provides status
        information. Printed in green when colours are available."""
        self.log(_REPORT_LEVEL, msg, *args, **kwargs)

    def tip(self, msg, *args, **kwargs):
        """Provide a more significant status statement (e.g. new section of the model). Similar to ``.info()``, but
        more emphasized. Printed in blue when colours are available."""
        self.log(_TIP_LEVEL, msg, *args, **kwargs)

# endregion


def init_root(root_name, stream_format=LogFormats.FANCY, log_debug=True):
    logging.addLevelName(_TIP_LEVEL, 'TIP')
    logging.addLevelName(_REPORT_LEVEL, 'REPORT')
    logging.setLoggerClass(ModelLogger)
    root_logger = logging.getLogger(root_name)
    root_logger.propagate = True
    if log_debug:
        root_logger.setLevel(1)
    else:
        root_logger.setLevel(logging.INFO)

    if isinstance(stream_format, str):
        try:
            stream_format = LogFormats[stream_format]
        except KeyError:
            pass

    if stream_format == LogFormats.FANCY:
        stdout_formatter = _prep_fancy_formatter()
    elif stream_format == LogFormats.BASIC:
        stdout_formatter = logging.Formatter(_FMT_STRING.format(arrow=_ASCII_ARROW))
    elif stream_format == LogFormats.JSON:
        stdout_formatter = _JsonFormatter()
    else:
        stdout_formatter = logging.Formatter(stream_format)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.addFilter(_RangeFilter(0, 100))
    stdout_handler.setFormatter(stdout_formatter)

    root_logger.handlers.clear()
    root_logger.addHandler(stdout_handler)

    return root_logger


def _prep_fancy_formatter():
    raw_fmt = _FMT_STRING.format(arrow=_UNC_ARROW)
    fmt_string = ''.join(["\x1b[{colour}m", raw_fmt, "\x1b[0m"])

    debug_formatter = logging.Formatter(fmt_string.format(colour=37))  # Grey colour
    info_formatter = logging.Formatter(fmt_string.format(colour=0))  # Default colour
    report_formatter = logging.Formatter(fmt_string.format(colour=32))  # Green colour
    tip_formatter = logging.Formatter(fmt_string.format(colour=34))  # Blue colour
    warn_formatter = logging.Formatter(fmt_string.format(colour=31))  # Red colour
    error_formatter = logging.Formatter(fmt_string.format(colour=41))  # Red BG colour
    critical_formatter = logging.Formatter(fmt_string.format(colour="1m\x1b[41"))  # Bold on red BG

    switch_formatter = _SwitchFormatter(raw_fmt, {
        logging.INFO: info_formatter, logging.WARNING: warn_formatter, _TIP_LEVEL: tip_formatter,
        _REPORT_LEVEL: report_formatter, logging.DEBUG: debug_formatter, logging.ERROR: error_formatter,
        logging.CRITICAL: critical_formatter
    })

    return switch_formatter


def get_model_logger(name):
    logging.setLoggerClass(ModelLogger)
    return logging.getLogger(name)


@contextmanager
def log_to_file(file_name: str, name, append=False, raw_ascii=False):
    root = logging.getLogger(name)

    write_mode = 'a' if append else 'w'
    handler = logging.FileHandler(file_name, mode=write_mode, encoding='utf-8')

    arrow = _ASCII_ARROW if raw_ascii else _UNC_ARROW
    fmt_str = _FMT_STRING.format(arrow=arrow)
    handler.setFormatter(logging.Formatter(fmt_str))

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
