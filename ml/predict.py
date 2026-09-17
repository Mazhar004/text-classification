import argparse
import os

import pandas as pd

# RASA
from rasa.shared.utils.cli import print_success

# Custom
import paths
from api_inference import collect_entities, sen_filter
from processing.custom.cli import configure_logging
from processing.custom.nlu_model import NluModel

COLUMNS = ["Query", "Intent", "Entities", "Confidence"]


def text_write(new_data, output_file):
    with open(output_file, 'w', encoding='utf-8') as fh:
        text_data = '\n\n'.join([', '.join(i) for i in new_data])
        fh.write(text_data + '\n')


def csv_write(new_data, output_file):
    df = pd.DataFrame(new_data, columns=COLUMNS)
    df.to_csv(output_file, index=False)


def run_inference(model_path, input_file, output_file):
    model = NluModel(model_path)

    print_success("NLU model loaded. Predicting queries from {}".format(
        input_file))

    new_data = []
    with open(input_file, 'r', encoding='utf-8') as fh:
        for line in fh:
            message = sen_filter(line)
            if message == "":
                continue

            result = model.parse(message)

            entities = '||'.join(
                '{}:{}'.format(name, value)
                for name, values in collect_entities(result).items()
                for value in values)
            intent = result.get('intent') or {}
            confidence = float(intent.get('confidence') or 0.0)

            new_data.append([message, intent.get('name'), entities,
                             str(round(confidence, 2))])

    new_data = sorted(new_data, key=lambda x: (x[1] or '', x[-1]))

    if str(output_file).lower().endswith('.csv'):
        csv_write(new_data, output_file)
    else:
        text_write(new_data, output_file)

    print_success("{} predictions written to {}".format(
        len(new_data), output_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default='assistant', required=False,
                        type=str, help="Name of the dataset")
    parser.add_argument("--inp", default='query_list.txt', required=False,
                        type=str, help="Path of the input file")
    parser.add_argument("--out", default='predict_list.txt', required=False,
                        type=str, help="Path of the output file")
    args = parser.parse_args()

    os.chdir(paths.ML_DIR)
    configure_logging()
    run_inference(paths.weight_dir(args.dataset),
                  paths.QUERY_DIR / args.inp,
                  paths.QUERY_DIR / args.out)
