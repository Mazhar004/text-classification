import asyncio
import argparse
import json
import os
import pandas as pd


# RASA
from rasa.cli.utils import print_success
from rasa.core.interpreter import INTENT_MESSAGE_PREFIX, RegexInterpreter
from rasa.nlu.model import Interpreter

# Custom
from processing.custom.get_model import get_model


def predict_json_form(ml_prediction):
    nlp_data = {}
    nlp_data['intent'] = ml_prediction['intent']['name']
    nlp_data['confidence'] = round(ml_prediction['intent']['confidence'], 2)
    nlp_data['entities'] = {}
    for i in ml_prediction["entities"]:
        try:
            nlp_data['entities'][i['entity']].append(i['value'])
        except:
            nlp_data['entities'][i['entity']] = [i['value']]
    return nlp_data


def text_write(new_data, output_file):
    with open(output_file, 'w') as fh:
        text_data = '\n\n'.join([', '.join(i) for i in new_data])
        fh.write(text_data)


def csv_write(new_data, output_file):
    df = pd.DataFrame(new_data, columns=[
                      "Query", "Intent", "Entities", "Confidence"])
    df.to_csv(output_file, index=False)


def run_inference(model_path, input_file, output_file):
    interpreter = Interpreter.load(model_path)
    regex_interpreter = RegexInterpreter()

    print_success(
        "NLU model loaded. Type a message and press enter to parse it.")
    nlp_data = {}
    with open(input_file, 'r') as fh:
        data = fh.readlines()
        new_data = []
        for i in data:
            message = i.lower().strip()
            if message == "":
                continue
            if message.startswith(INTENT_MESSAGE_PREFIX):
                loop = asyncio.get_event_loop()
                result = loop.run_until_complete(
                    regex_interpreter.parse(message))
            else:
                result = interpreter.parse(message)

            output = {key: result[key]
                      for key in result.keys() & {'intent', 'entities'}}

            nlp_data[message] = predict_json_form(output)

            entities = '||'.join(([i['entity'] + ':' + i['value']
                                   for i in output["entities"]]))
            intent = output['intent']['name']
            confidence = output['intent']['confidence']
            new_data.append([message, intent, entities, str(
                round(confidence, 2))])
        new_data = sorted(new_data, key=lambda x: (x[1], x[-1]))

        if '.csv' in output_file.lower():
            csv_write(new_data, output_file)
        else:
            text_write(new_data, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default='assistant', required=False,
                        type=str, help="Name of the dataset")
    parser.add_argument("--inp", default='query_list.txt', required=False,
                        type=str, help="Path of the input file")
    parser.add_argument("--out", default='predict_list.txt', required=False,
                        type=str, help="Path of the output file")
    args = parser.parse_args()
    args.inp = 'query_predict/' + args.inp
    args.out = 'query_predict/' + args.out

    model_path = 'model_files/model_weight/' + args.dataset
    model_folder = get_model(model_path)
    run_inference(model_folder, args.inp, args.out)
