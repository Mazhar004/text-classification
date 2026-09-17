import argparse
import os

# Custom
import paths
from processing.custom.cli import add_split_argument
from processing.custom.evaluate import evaluate


def custom_test(dataset, split):
    nlu_data = paths.test_data_file(
        dataset) if split else paths.nlu_file(dataset)
    if not nlu_data.exists():
        raise FileNotFoundError(
            'Evaluation data not found: {}\nRun train.py{} first.'.format(
                nlu_data, ' --split' if split else ''))

    evaluate(paths.weight_dir(dataset), nlu_data,
             paths.performance_dir(dataset))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default='assistant', required=False,
                        type=str, help="Name of the dataset")
    add_split_argument(parser, "Evaluate on the held-out test split "
                               "instead of the full dataset")
    args = parser.parse_args()

    os.chdir(paths.ML_DIR)
    custom_test(args.dataset, args.split)
