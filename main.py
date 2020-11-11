import os

# Flask
from flask_cors import CORS
from flask import Flask, request

# RASA
from rasa.nlu.model import Interpreter
from rasa.core.interpreter import RegexInterpreter

# Custom
from ml.api_inference import chat
from ml.processing.custom.get_model import get_model


app = Flask(__name__)
cors = CORS(app)

dataset_name = 'assistant'
model_path = 'ml/model_files/model_weight/' + dataset_name
model_folder = get_model(model_path)
interpreter = Interpreter.load(model_folder)
regex_interpreter = RegexInterpreter()


@app.route('/')
def index():
    return 'Hello'


@app.route('/chat/<string>')
def process(string):
    user_input = {"user_input": string}
    nlp_data = chat(interpreter, regex_interpreter, user_input)

    return nlp_data


if __name__ == '__main__':
    app.run(debug=True, host='localhost')
