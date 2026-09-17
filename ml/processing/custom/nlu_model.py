"""Thin wrapper around a trained Rasa NLU model.

Rasa 3.x removed ``rasa.nlu.model.Interpreter``; messages are now parsed
through an ``Agent``. ``Agent.parse_message`` is a coroutine and also handles
``/intent`` payloads itself, so the separate ``RegexInterpreter`` that the
Rasa 1.x code needed is no longer required.
"""

import asyncio
import threading

from rasa.core.agent import Agent

from .get_model import get_model


class NluModel():
    """Loads a model once and parses messages against it."""

    def __init__(self, model_path):
        self.model_path = get_model(model_path)
        self.agent = Agent.load(self.model_path)
        # The Flask dev server is threaded and a Rasa processor is not
        # guaranteed to be re-entrant, so parsing is serialised.
        self._lock = threading.Lock()

    def parse(self, message):
        """Parse one message and return Rasa's raw prediction dict."""
        with self._lock:
            return asyncio.run(self.agent.parse_message(message))
