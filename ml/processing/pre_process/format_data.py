import pandas as pd
import os
import json


class DataLoad():
    csv_folder = 'model_files/model_dataset/csv/'
    md_folder = 'model_files/model_dataset/data/'

    def __init__(self, dataset):
        try:
            self.dataset = DataLoad.csv_folder+dataset+'.csv'
            self.all_intent, self.all_query = self.fileopen()
            self.json_data = self.data_format()
            self.rasa_save(DataLoad.md_folder+dataset+'/')
            print("Data formatting Completed!")
        except Exception as error:
            print("Error happend: {}".format(error))

    def fileopen(self):
        df = pd.read_csv(self.dataset)
        all_intent = df['Intent']
        all_query = df['Qus']
        return all_intent, all_query

    def sen_filter(self, query):
        space = '/\\,?!#.|;"'
        no_space = "'-"
        for i in space:
            query = query.replace(i, ' ')
        for i in no_space:
            query = query.replace(i, '')
        return query.lower().strip()

    def qus_process(self, query_set):
        processed_query = []
        for i in query_set.split('\n'):
            processed_query.append(self.sen_filter(i))
        return processed_query

    def intent_process(self, intent):
        return intent.strip()

    def data_format(self):
        json_data = {}
        for i, j in zip(self.all_intent, self.all_query):
            processed_intent = self.intent_process(i)
            processed_query = self.qus_process(j)
            json_data[processed_intent] = processed_query
        return json_data

    def __getitem__(self, key):
        try:
            return self.json_data[key]
        except Exception as Err:
            return Err

    def rasa_save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(path+'nlu.md', 'w') as fh:
            for i, j in self.json_data.items():
                intent = '## intent:{}\n'.format(i)
                qus = '\n'.join(['- '+m for m in j if m])
                fh.write(intent+qus)
                fh.write('\n\n')

    def __str__(self):
        return json.dumps(self.json_data, indent=2)
