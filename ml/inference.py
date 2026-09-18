import argparse
import os

# RASA
from rasa.shared.utils.cli import print_success

# Custom
import paths
from api_inference import collect_entities
from processing.custom.cli import configure_logging
from processing.custom.nlu_model import NluModel


def run_inference(model_path):
    model = NluModel(model_path)

    print_success(
        "NLU model loaded. Type a message and press enter to parse it.")

    while True:
        print_success("User:")
        try:
            message = input().strip()
        except EOFError:
            break
        if message == "":
            break

        result = model.parse(message)

        print((result.get('intent') or {}).get('name'))
        print(','.join('{}:{}'.format(name, value)
                       for name, values in collect_entities(result).items()
                       for value in values))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='assistant', required=False,
                        type=str, help="Name of the dataset")
    args = parser.parse_args()

    os.chdir(paths.ML_DIR)
    configure_logging()
    run_inference(paths.weight_dir(args.dataset))
