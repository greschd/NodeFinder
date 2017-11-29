import sys
import logging

_LOGGER_NAME = __name__.split('.')[0]
_DEFAULT_HANDLER = logging.StreamHandler(sys.stdout)
_DEFAULT_HANDLER.setFormatter(
    logging.Formatter(style='{', fmt='{levelname} - {name} - {message}')
)
_LOGGER = logging.getLogger(_LOGGER_NAME)
_LOGGER.setLevel(logging.WARNING)
_LOGGER.addHandler(_DEFAULT_HANDLER)
