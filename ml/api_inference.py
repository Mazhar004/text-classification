import asyncio

from rasa.core.interpreter import INTENT_MESSAGE_PREFIX
from rasa.nlu.model import Interpreter


def sen_filter(query):
    space = '/\\,?!#.|;"(){}[]<>+-=*^%'
    no_space = "'-"
    for i in space:
        query = query.replace(i, ' ')
    for i in no_space:
        query = query.replace(i, '')
    return query.lower().strip()


def rasa_ml(interpreter, regex_interpreter, sentence):
    if sentence.startswith(INTENT_MESSAGE_PREFIX):
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(regex_interpreter.parse(sentence))
    else:
        result = interpreter.parse(sentence)

    ml_prediction = {key: result[key]
                     for key in result.keys() & {'intent', 'entities'}}

    return ml_prediction


def predict_json_form(sentence, ml_prediction, threshold=0.5):
    nlp_data = {}
    nlp_data['sentence'] = sentence
    nlp_data['intent'] = ml_prediction['intent']['name']
    nlp_data['confidence'] = ml_prediction['intent']['confidence']
    nlp_data['threshold'] = threshold
    if nlp_data['confidence'] >= threshold:
        nlp_data['status'] = True
    else:
        nlp_data['status'] = False
    nlp_data['entities'] = {}
    for i in ml_prediction["entities"]:
        try:
            nlp_data['entities'][i['entity']].append(i['value'])
        except:
            nlp_data['entities'][i['entity']] = [i['value']]
    return nlp_data


def chat(interpreter, regex_interpreter, inp):
    sentence = sen_filter(inp['user_input'])
    ml_prediction = rasa_ml(interpreter, regex_interpreter, sentence)
    nlp_data = predict_json_form(sentence, ml_prediction)

    return nlp_data
