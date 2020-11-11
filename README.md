# Text Classification

## Installation

### Python Version

- Python == 3.7 (Required)

### Library Installation

#### Windows

- Virtual Environment
  - `python -m venv venv`
  - `.\venv\Scripts\activate`
  - If any problem for scripts activation
    - Execute following command in administration mode
      - `Set-ExecutionPolicy Unrestricted -Force`
    - Later you can revert the change
      - `Set-ExecutionPolicy restricted -Force`
- Library Install
  - `python .\installation\get-pip.py`
  - `pip install --upgrade pip`
  - `pip install --upgrade setuptools`
  - `pip install -r requirements.txt`
  - `python -m spacy download en_core_web_md`
  - Open Command Window as Administrator
    - Activate Virtual Environment
    - `python -m spacy link en_core_web_md en`

#### Linux

- Virtual Environment
  - `python -m venv venv`
  - `source venv/bin/activate`
- Library Install
  - `pip install --upgrade pip`
  - `pip install --upgrade setuptools`
  - `pip install -r requirements.txt`
  - `python -m spacy download en_core_web_md`
  - `python -m spacy link en_core_web_md en`

### Code Dtructure

```
    ml ---|
          |---model_files----|
          |                  |-----model_config------|

          |                  |-----model_dataset-----|
          |                  |                       |-----csv-------|
          |                  |                       |-----data------|
          |                  |                       |               |-----assistant-----|
          |                  |                       |               |                   |-----train_test_split-----|
          |                  |                       |               |                   |----------nlu.md----------|

          |                  |---model_performance---|               |
          |                  |                       |---assistant---|

          |                  |-----model_weight------|               |
          |                  |                       |---assistant---|
```

### Machine Learning Model

- `cd ml`

#### Train

- Parameter
  - --dataset [Name of the dataset same as csv file name]
  - --split [Default Value **False**]
    - if **True** splitted the data then train & test
    - if **False** train the model with full data
- Train with full data
  - `python train.py --dataset [Dataset_Name]`
- Train with splitted data
  - `python train.py --dataset [Dataset_Name] --split True`
- Example:
  - `python train.py --dataset assistant`
  - `python train.py --dataset assistant --split True`

#### Test

- Parameter
  - --dataset [Name of the dataset same as csv file name]
  - --split [Default Value **False**]
    - if **True** then splitted data & evaluate the model
    - if **False** evaluate the model with full data
- Test with full data
  - `python test.py --dataset [Dataset_Name]`
- Test with splitted data
  - `python test.py --dataset [Dataset_Name] --split True`
- Example:
  - `python test.py --dataset assistant`

#### Inference

- Command line query testing
- Parameter
  - --dataset [Name of the dataset same as csv file name]
- Inference file run
  - `python inference.py --dataset [Dataset_Name]`
- Example:
  - `python inference.py --dataset assistant`

#### Predict

- Test query list from a text file, in which each query separated by new line

- Parameter
  - --dataset [Name of the dataset same as csv file name]
  - --inp [Input file name]
  - --out [Output file name. If output file name have .csv extension it will generate **csv** file other wise **text** file ]
- Predict file run
  - `python predict.py --dataset dataset_name --inp [Input_Filename] --out [Output_Filename]`
- Example:
  - Text file generate:
    - `python predict.py --dataset assistant --inp query_list.txt --out predict_list.txt`
  - CSV file generate
    - `python predict.py --dataset assistant --inp query_list.txt --out predict_list.csv`

#### Backup & Restore

- Backup the model data,configuration,weight
- Restore the model data,configuration,weight

- Parameter
  - --dataset [Name of the dataset same as csv file name]
  - --foldername [In case of model restore, need to declare folder name where backup saved]
- Backup file run
  - `python backup.py --dataset dataset_name`
- Restore file run
  - `python backup.py --dataset dataset_name --foldername [Folder_Name]`
- Example:
  - Backup:
    - `python backup.py --dataset assistant`
  - Restore
    - `python backup.py --dataset assistant --foldername assistant_10_Nov_10_56_44_PM`

### Log Data

- `tensorboard --logdir .\model_files\model_weight\tensorboard\[Dataset_Name]`
- Example
  - `tensorboard --logdir .\model_files\model_weight\tensorboard\assistant`

### API Run

- `python main.py`
