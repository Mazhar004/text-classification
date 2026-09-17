"""Convert the CSV dataset into Rasa YAML NLU training data.

Rasa 3.x removed the Markdown training-data format, so this module writes
``nlu.yml`` in the ``3.1`` YAML schema instead of the old ``nlu.md``.
"""

import json
import re

import pandas as pd

INTENT_COLUMN = 'Intent'
QUERY_COLUMN = 'Qus'
TRAINING_DATA_VERSION = '3.1'

# Characters replaced by a space, and characters removed outright.
SPACE_CHARS = '/\\,?!#.|;"'
STRIP_CHARS = "'-"

# An intent name that can be written as a bare YAML scalar.
PLAIN_NAME = re.compile(r'^[A-Za-z0-9_][A-Za-z0-9_+.-]*$')


def yaml_name(name):
    """Quote an intent name unless it is safe as a plain YAML scalar."""
    return name if PLAIN_NAME.match(name) else json.dumps(name)


class DataLoad():
    """Read ``<dataset>.csv`` and persist it as Rasa YAML training data."""

    def __init__(self, csv_path, out_dir):
        self.csv_path = csv_path
        self.out_dir = out_dir
        self.all_intent, self.all_query = self.fileopen()
        self.json_data = self.data_format()
        self.rasa_save(out_dir)
        print('Data formatting completed: {} intents -> {}'.format(
            len(self.json_data), out_dir / 'nlu.yml'))

    def fileopen(self):
        df = pd.read_csv(self.csv_path)
        missing = {INTENT_COLUMN, QUERY_COLUMN} - set(df.columns)
        if missing:
            raise ValueError('{} is missing the column(s): {}'.format(
                self.csv_path, ', '.join(sorted(missing))))
        df = df.dropna(subset=[INTENT_COLUMN, QUERY_COLUMN])
        return df[INTENT_COLUMN], df[QUERY_COLUMN]

    def sen_filter(self, query):
        for i in SPACE_CHARS:
            query = query.replace(i, ' ')
        for i in STRIP_CHARS:
            query = query.replace(i, '')
        return query.lower().strip()

    def qus_process(self, query_set):
        """Split one cell into its individual, de-duplicated examples."""
        processed_query = []
        for i in str(query_set).split('\n'):
            example = self.sen_filter(i)
            if example and example not in processed_query:
                processed_query.append(example)
        return processed_query

    def intent_process(self, intent):
        return str(intent).strip()

    def data_format(self):
        json_data = {}
        for i, j in zip(self.all_intent, self.all_query):
            processed_intent = self.intent_process(i)
            if not processed_intent:
                continue
            # The same intent may appear on several CSV rows.
            examples = json_data.setdefault(processed_intent, [])
            for example in self.qus_process(j):
                if example not in examples:
                    examples.append(example)
        return json_data

    def __getitem__(self, key):
        return self.json_data[key]

    def rasa_save(self, path):
        path.mkdir(parents=True, exist_ok=True)
        lines = ['version: "{}"'.format(TRAINING_DATA_VERSION), '', 'nlu:']
        for intent, examples in self.json_data.items():
            if not examples:
                print('Skipping intent "{}": no usable examples.'.format(intent))
                continue
            lines.append('- intent: {}'.format(yaml_name(intent)))
            lines.append('  examples: |')
            lines.extend('    - {}'.format(example) for example in examples)
            lines.append('')
        with open(path / 'nlu.yml', 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(lines) + '\n')

    def __str__(self):
        return json.dumps(self.json_data, indent=2)
