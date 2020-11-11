from datetime import datetime as dt
import argparse
import shutil
import os


class DataStore():
    def __init__(self, dataset, foldername=""):
        self.dataset = dataset
        self.foldername = foldername
        self.time = dt.now().strftime("_%d_%b_%I_%M_%S_%p")
        self.existed_folder = self.old_folder()
        self.new_version = self.new_folder()
        if self.foldername == "":
            self.backup()
        else:
            self.restore()

    def old_folder(self):
        csv_sheet = 'model_files/model_dataset/csv/'
        data_folder = 'model_files/model_dataset/data/' + self.dataset
        config_folder = 'model_files/model_config/' + self.dataset
        weight = 'model_files/model_weight/' + self.dataset
        result = 'model_files/model_performance/' + self.dataset

        return [csv_sheet, data_folder, config_folder, weight, result]

    def new_folder(self):
        version_path = 'model_files/previous_version/'
        if self.foldername == "":
            bot_version = version_path + self.dataset + self.time + '/'
        else:
            bot_version = version_path + self.foldername + '/'
        bot_version_csv = bot_version + 'data/'
        bot_version_nlu = bot_version+'data/data'
        bot_version_config = bot_version+'config/'
        bot_version_weight = bot_version+'weight/'
        bot_version_result = bot_version + 'result/'

        os.makedirs(version_path, exist_ok=True)
        os.makedirs(bot_version, exist_ok=True)
        os.makedirs(bot_version_csv, exist_ok=True)

        return [bot_version_csv, bot_version_nlu, bot_version_config, bot_version_weight, bot_version_result]

    def backup(self):
        shutil.copy(self.existed_folder[0]+self.dataset +
                    '.csv', self.new_version[0] + self.dataset + '.csv')
        for i, j in zip(self.existed_folder[1:], self.new_version[1:]):
            shutil.copytree(i, j)

    def restore(self):
        for i in self.existed_folder:
            try:
                shutil.rmtree(i)
            except:
                pass
        os.makedirs(self.existed_folder[0], exist_ok=True)

        shutil.copy(self.new_version[0]+self.dataset+'.csv',
                    self.existed_folder[0]+self.dataset + '.csv')
        for i, j in zip(self.new_version[1:], self.existed_folder[1:]):
            shutil.copytree(i, j)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default='assistant', required=False,
                        type=str, help="Name of the dataset")
    parser.add_argument("--foldername", default="", required=False,
                        type=str, help="For restore type foldername")
    args = parser.parse_args()
    datastore = DataStore(args.dataset, args.foldername)
