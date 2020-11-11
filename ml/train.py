import argparse
import shutil
import os

# RASA
from rasa.nlu import load_data
from rasa.nlu.model import Trainer
from rasa.nlu import config
from rasa.nlu import test

# Custom
from processing.pre_process.format_data import DataLoad


def train(dataset, split):
    if split:
        training_data = load_data(
            'model_files/model_dataset/data/'+dataset+'/train_test_split/training_data.md')
    else:
        training_data = load_data(
            'model_files/model_dataset/data/'+dataset+'/nlu.md')
    trainer = Trainer(config.load(
        "model_files/model_config/"+dataset+"/config.yml"))
    trainer.train(training_data)
    try:
        shutil.rmtree('model_files/model_weight/' + dataset)
    except:
        pass

    model_directory = trainer.persist('model_files/model_weight/' + dataset)
    temp = model_directory.split('/')
    if len(temp) > 1:
        model_folder = temp[-1]
    else:
        temp = model_directory.split('\\')
        model_folder = temp[-1]
    with open('model_files/model_weight/'+dataset+'/latest_model_path.txt', 'w') as fh:
        fh.write(model_folder)

    if split:
        data_path = 'model_files/model_dataset/data/' + \
            dataset+'/train_test_split/test_data.md'
        output_path = 'model_files/model_performance/' + dataset
        test(data_path, model_directory, output_path, successes=True,
             errors=True, confmat='confmat.png', histogram='hist.png')


def data_prepare(dataset):
    try:
        shutil.rmtree('model_files/model_dataset/data/' + dataset)
    except:
        pass
    DataLoad(dataset)
    string = 'rasa data split nlu -u model_files/model_dataset/data/{} --out model_files/model_dataset/data/{}/train_test_split'.format(
        dataset, dataset)
    os.system(string)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default='assistant', required=False,
                        type=str, help="Name of the dataset")
    parser.add_argument("--split", default=False, required=False,
                        type=bool, help="Data split")
    args = parser.parse_args()
    data_prepare(args.dataset)
    train(args.dataset, args.split)
