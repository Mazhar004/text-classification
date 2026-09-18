"""Shared CLI helpers."""

import argparse
import os

TRUE_VALUES = ('true', 't', 'yes', 'y', '1')
FALSE_VALUES = ('false', 'f', 'no', 'n', '0')


def str2bool(value):
    """Parse a boolean flag value.

    ``type=bool`` cannot be used for this: argparse would hand it the raw
    string, and ``bool("False")`` is ``True``.
    """
    if isinstance(value, bool):
        return value
    if value.lower() in TRUE_VALUES:
        return True
    if value.lower() in FALSE_VALUES:
        return False
    raise argparse.ArgumentTypeError(
        'expected a boolean value, got {!r}'.format(value))


def add_split_argument(parser, help_text):
    """Add a ``--split`` flag accepting both ``--split`` and ``--split True``."""
    parser.add_argument("--split", nargs='?', const=True, default=False,
                        type=str2bool, help=help_text)


def configure_logging(default='WARNING'):
    """Quiet Rasa's logging for the inference entry points.

    Importing Rasa directly (rather than going through its CLI) leaves the
    structlog stream at debug, which prints a ``processor.message.parse`` line
    for every parsed message. ``LOG_LEVEL`` overrides this for debugging.
    """
    import logging

    from rasa.utils.common import configure_logging_and_warnings
    from rasa.utils.log_utils import configure_structlog

    name = os.environ.get('LOG_LEVEL', default).upper()
    level = getattr(logging, name, None)
    if not isinstance(level, int):
        level = logging.WARNING

    configure_logging_and_warnings(level)
    # Rasa's structlog stream is configured separately from stdlib logging;
    # without this the per-message parse events still reach stdout.
    configure_structlog(level)
