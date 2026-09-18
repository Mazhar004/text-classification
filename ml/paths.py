"""Filesystem layout of the project.

Every path is derived from this file's own location, so the scripts in ``ml/``
work no matter which directory they are launched from.
"""

from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ML_DIR.parent

MODEL_FILES = ML_DIR / 'model_files'
CSV_DIR = MODEL_FILES / 'model_dataset' / 'csv'
DATA_DIR = MODEL_FILES / 'model_dataset' / 'data'
CONFIG_DIR = MODEL_FILES / 'model_config'
WEIGHT_DIR = MODEL_FILES / 'model_weight'
PERFORMANCE_DIR = MODEL_FILES / 'model_performance'
VERSION_DIR = MODEL_FILES / 'previous_version'
QUERY_DIR = ML_DIR / 'query_predict'

# Rasa 3.x dropped the Markdown training-data format in favour of YAML.
NLU_FILENAME = 'nlu.yml'
SPLIT_DIRNAME = 'train_test_split'


def csv_file(dataset):
    return CSV_DIR / '{}.csv'.format(dataset)


def data_dir(dataset):
    return DATA_DIR / dataset


def nlu_file(dataset):
    return data_dir(dataset) / NLU_FILENAME


def split_dir(dataset):
    return data_dir(dataset) / SPLIT_DIRNAME


def test_data_file(dataset):
    return split_dir(dataset) / 'test_data.yml'


def training_data_file(dataset):
    return split_dir(dataset) / 'training_data.yml'


def config_file(dataset):
    return CONFIG_DIR / dataset / 'config.yml'


def weight_dir(dataset):
    return WEIGHT_DIR / dataset


def performance_dir(dataset):
    return PERFORMANCE_DIR / dataset
