import argparse
import os
import shutil
import subprocess
import sys

# RASA
from rasa.model_training import train_nlu

# Custom
import paths
from processing.custom.cli import add_split_argument
from processing.custom.evaluate import evaluate
from processing.pre_process.format_data import DataLoad


def data_prepare(dataset, split):
    """Rebuild the YAML training data from the CSV, optionally splitting it."""
    data_dir = paths.data_dir(dataset)
    shutil.rmtree(data_dir, ignore_errors=True)

    DataLoad(paths.csv_file(dataset), data_dir)

    if split:
        # A list argument (rather than os.system with an interpolated string)
        # keeps a dataset name from being interpreted by the shell.
        subprocess.run(
            [sys.executable, '-m', 'rasa', 'data', 'split', 'nlu',
             '--nlu', str(paths.nlu_file(dataset)),
             '--out', str(paths.split_dir(dataset))],
            check=True)


def train(dataset, split):
    nlu_data = paths.training_data_file(
        dataset) if split else paths.nlu_file(dataset)
    if not nlu_data.exists():
        raise FileNotFoundError('Training data not found: {}'.format(nlu_data))

    config = paths.config_file(dataset)
    if not config.exists():
        raise FileNotFoundError('Pipeline config not found: {}'.format(config))

    output = paths.weight_dir(dataset)
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True, exist_ok=True)

    model = train_nlu(
        config=str(config), nlu_data=str(nlu_data), output=str(output))
    if model is None:
        raise RuntimeError('Training failed; see the Rasa output above.')
    print('Model saved to {}'.format(model))

    if split:
        evaluate(model, paths.test_data_file(dataset),
                 paths.performance_dir(dataset))

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default='assistant', required=False,
                        type=str, help="Name of the dataset")
    add_split_argument(parser, "Split the data, train on the training half "
                               "and evaluate on the held-out half")
    args = parser.parse_args()

    # config.yml points TensorBoard at a relative directory, so anchor the
    # process to ml/ and the logs land in the same place however it was called.
    os.chdir(paths.ML_DIR)

    data_prepare(args.dataset, args.split)
    train(args.dataset, args.split)
