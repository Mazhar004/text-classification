import asyncio
import argparse
import os


# RASA
from rasa.cli.utils import print_success
from rasa.core.interpreter import INTENT_MESSAGE_PREFIX, RegexInterpreter
from rasa.nlu.model import Interpreter

# Custom
from processing.custom.get_model import get_model


def run_inference(model_path):
    interpreter = Interpreter.load(model_path)
    regex_interpreter = RegexInterpreter()

    print_success(
        "NLU model loaded. Type a message and press enter to parse it.")

    while True:
        print_success("User:")
        message = input().strip()
        if message == "":
            break
        if message.startswith(INTENT_MESSAGE_PREFIX):
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(regex_interpreter.parse(message))
        else:
            result = interpreter.parse(message)

        output = {key: result[key]
                  for key in result.keys() & {'intent', 'entities'}}
        entities = ','.join(([i['entity'] + ':' + i['value']
                              for i in output["entities"]]))
        intent = output['intent']['name']
        print(intent)
        print(entities)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='assistant', required=False,
                        type=str, help="Name of the dataset")
    args = parser.parse_args()
    model_path = 'model_files/model_weight/' + args.dataset
    model_folder = get_model(model_path)
    run_inference(model_folder)
