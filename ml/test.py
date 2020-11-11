import argparse
import os

# RASA
from rasa.nlu import test

# Custom
from processing.custom.get_model import get_model


def custom_test(dataset, split):
    if split:
        data_path = 'model_files/model_dataset/data/' + \
            dataset+'/train_test_split/test_data.md'
    else:
        data_path = 'model_files/model_dataset/data/'+dataset
    model_path = 'model_files/model_weight/' + dataset
    output_path = 'model_files/model_performance/' + dataset

    os.makedirs('model_files/model_performance/', exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    model_folder = get_model(model_path)

    test(data_path, model_folder, output_path, successes=True,
         errors=True, confmat='confmat.png', histogram='hist.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default='assistant', required=False,
                        type=str, help="Name of the dataset")
    parser.add_argument("--split", default=False, required=False,
                        type=bool, help="Data split")
    args = parser.parse_args()
    custom_test(args.dataset, args.split)
