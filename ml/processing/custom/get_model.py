"""Locate the trained model archive for a dataset.

Rasa 3.x persists a model as a single ``.tar.gz`` archive, so the
``latest_model_path.txt`` pointer used by the Rasa 1.x layout is gone --
the newest archive in the directory is discovered directly.
"""

import os

from rasa.model import get_latest_model


class ModelNotFound(FileNotFoundError):
    """Raised when a dataset has no trained model yet."""


def get_model(model_path):
    """Return the path of the model archive to load.

    ``model_path`` may point either at a ``.tar.gz`` archive or at a directory
    holding one or more of them, in which case the most recent one is used.
    """
    model_path = str(model_path)

    if os.path.isfile(model_path):
        return model_path

    latest = get_latest_model(model_path) if os.path.isdir(model_path) else None
    if not latest:
        raise ModelNotFound(
            'No trained model found in "{}". Train one first with:\n'
            '    python ml/train.py --dataset <dataset_name>'.format(model_path))
    return latest
