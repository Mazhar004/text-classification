"""Text normalisation and response shaping shared by the CLI tools and the API.

This module deliberately imports nothing from the project so that it can be
loaded both as ``ml.api_inference`` (from ``main.py`` at the repository root)
and as ``api_inference`` (from the scripts inside ``ml/``).
"""

# Characters replaced by a space, and characters removed outright.
SPACE_CHARS = '/\\,?!#.|;"(){}[]<>+-=*^%'
STRIP_CHARS = "'-"

# Messages starting with this are Rasa intent payloads (e.g. ``/greet``) and
# must reach the model untouched -- normalising them would strip the prefix.
INTENT_MESSAGE_PREFIX = '/'

DEFAULT_THRESHOLD = 0.5


def sen_filter(query):
    """Normalise free-text user input before it reaches the model."""
    if query.startswith(INTENT_MESSAGE_PREFIX):
        return query.strip()
    for i in SPACE_CHARS:
        query = query.replace(i, ' ')
    for i in STRIP_CHARS:
        query = query.replace(i, '')
    return query.lower().strip()


def collect_entities(prediction):
    """Group Rasa's flat entity list into ``{entity_name: [values]}``."""
    entities = {}
    for item in prediction.get('entities') or []:
        entities.setdefault(item['entity'], []).append(str(item['value']))
    return entities


def predict_json_form(sentence, prediction, threshold=DEFAULT_THRESHOLD):
    intent = prediction.get('intent') or {}
    confidence = float(intent.get('confidence') or 0.0)

    return {
        'sentence': sentence,
        'intent': intent.get('name'),
        'confidence': confidence,
        'threshold': threshold,
        'status': confidence >= threshold,
        'entities': collect_entities(prediction),
    }


def chat(model, user_input, threshold=DEFAULT_THRESHOLD):
    """Normalise, parse and format a single user message."""
    sentence = sen_filter(user_input)
    prediction = model.parse(sentence)

    return predict_json_form(sentence, prediction, threshold)
