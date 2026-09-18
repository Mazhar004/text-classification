"""HTTP API around the trained Rasa NLU model.

Configuration is read from the environment:

    DATASET               dataset/model name to serve      (default: assistant)
    CORS_ORIGINS          comma-separated allowed origins  (default: disabled)
    CONFIDENCE_THRESHOLD  score at or above which `status` is true  (default: 0.5)
    MAX_INPUT_LENGTH      longest accepted message         (default: 1000)
    HOST / PORT           dev-server bind address     (default: 127.0.0.1:5000)
    FLASK_DEBUG           enable the debugger              (default: off)
    LOG_LEVEL             Rasa/root log level              (default: INFO)
"""

import os
import threading

# Flask
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

# Custom
from ml.api_inference import DEFAULT_THRESHOLD, chat
from ml.paths import weight_dir
from ml.processing.custom.cli import configure_logging
from ml.processing.custom.nlu_model import NluModel

TRUE_VALUES = ('1', 'true', 't', 'yes', 'y', 'on')

DATASET = os.environ.get('DATASET', 'assistant')
THRESHOLD = float(os.environ.get('CONFIDENCE_THRESHOLD', DEFAULT_THRESHOLD))
MAX_INPUT_LENGTH = int(os.environ.get('MAX_INPUT_LENGTH', '1000'))

_model = None
_model_lock = threading.Lock()


def env_flag(name, default=False):
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in TRUE_VALUES


def get_nlu_model():
    """Load the model on first use and reuse it afterwards."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = NluModel(weight_dir(DATASET))
    return _model


def create_app():
    configure_logging('INFO')

    app = Flask(__name__)

    # CORS is opt-in. Leaving CORS_ORIGINS unset keeps the API same-origin
    # only; the previous `CORS(app)` echoed every origin back, which is what
    # the Flask-Cors advisories (CVE-2024-6221 and friends) are about.
    origins = [o.strip()
               for o in os.environ.get('CORS_ORIGINS', '').split(',')
               if o.strip()]
    if origins:
        CORS(app, resources={r'/chat.*': {'origins': origins}},
             methods=['GET', 'POST'])

    @app.get('/')
    def index():
        return jsonify({'service': 'text-classification',
                        'dataset': DATASET,
                        'endpoints': ['/health', '/chat (POST)', '/chat/<text>']})

    @app.get('/health')
    def health():
        try:
            model = get_nlu_model()
        except Exception as error:  # model missing or unreadable
            return jsonify({'status': 'unavailable',
                            'detail': str(error)}), 503
        return jsonify({'status': 'ok', 'model': str(model.model_path)})

    def classify(user_input):
        if not isinstance(user_input, str) or not user_input.strip():
            return jsonify({'error': 'A non-empty "user_input" is required.'}), 400
        if len(user_input) > MAX_INPUT_LENGTH:
            return jsonify({
                'error': 'Input exceeds {} characters.'.format(MAX_INPUT_LENGTH),
            }), 413

        return jsonify(chat(get_nlu_model(), user_input, THRESHOLD))

    @app.post('/chat')
    def process_json():
        payload = request.get_json(silent=True) or {}
        return classify(payload.get('user_input'))

    @app.get('/chat/<path:text>')
    def process(text):
        """Kept for backwards compatibility; prefer POST /chat.

        Query text in a URL ends up in access logs and browser history.
        """
        return classify(text)

    @app.errorhandler(FileNotFoundError)
    def handle_missing_model(error):
        return jsonify({'error': str(error)}), 503

    @app.errorhandler(Exception)
    def handle_unexpected(error):
        # Flask routes HTTPExceptions here too, so 404/405 must pass through
        # rather than being reported as an internal error.
        if isinstance(error, HTTPException):
            return jsonify({'error': error.description}), error.code
        app.logger.exception('Unhandled error while classifying')
        return jsonify({'error': 'Internal server error.'}), 500

    return app


app = create_app()


if __name__ == '__main__':
    # Fail fast with a clear message instead of on the first request.
    get_nlu_model()

    # Flask's development server is not meant for production; run it behind a
    # WSGI server (e.g. `gunicorn 'main:app'`) when deploying.
    app.run(host=os.environ.get('HOST', '127.0.0.1'),
            port=int(os.environ.get('PORT', '5000')),
            debug=env_flag('FLASK_DEBUG'))
