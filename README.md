# Text Classification

Intent classification and entity extraction built on [Rasa](https://rasa.com/) NLU,
with a small Flask API in front of the trained model.

## Installation

### Python Version

- Python >= 3.8, < 3.11 (**3.10 recommended**)
  - The upper bound comes from Rasa 3.6, the latest Apache-2.0 Rasa Open Source release.

### Library Installation

#### Windows

- Virtual Environment
  - `py -3.10 -m venv venv`
  - `.\venv\Scripts\activate`
  - If script activation is blocked
    - Execute the following in an administrator shell
      - `Set-ExecutionPolicy Unrestricted -Force`
    - Later you can revert the change
      - `Set-ExecutionPolicy restricted -Force`
- Library Install
  - `python -m pip install --upgrade pip setuptools`
  - `pip install -r requirements.txt`
  - `python -m spacy download en_core_web_md`

#### Linux / macOS

- Virtual Environment
  - `python3.10 -m venv venv`
  - `source venv/bin/activate`
- Library Install
  - `python -m pip install --upgrade pip setuptools`
  - `pip install -r requirements.txt`
  - `python -m spacy download en_core_web_md`

> `spacy link` no longer exists in spaCy 3. The model is referenced by its real
> name (`en_core_web_md`) in `config.yml`, so no aliasing step is needed.

### Code Structure

```
    ml ---|
          |---model_files----|
          |                  |-----model_config------|

          |                  |-----model_dataset-----|
          |                  |                       |-----csv-------|
          |                  |                       |-----data------|
          |                  |                       |               |-----assistant-----|
          |                  |                       |               |                   |-----train_test_split-----|
          |                  |                       |               |                   |----------nlu.yml---------|

          |                  |---model_performance---|               |
          |                  |                       |---assistant---|

          |                  |-----model_weight------|               |
          |                  |                       |---assistant---|
```

Training data is generated from `model_files/model_dataset/csv/<dataset>.csv`
into `nlu.yml` (Rasa 3.x YAML format; the old `nlu.md` format was removed in Rasa 3).

## Machine Learning Model

The scripts resolve their own paths, so they can be run from anywhere:

```bash
python ml/train.py --dataset assistant
```

`cd ml && python train.py --dataset assistant` works exactly the same.

### Train

- Parameter
  - `--dataset` [Name of the dataset, same as the csv file name]
  - `--split` [Default **False**]
    - if **True**, split the data, train on the training half and evaluate on the held-out half
    - if **False**, train the model with the full data
- Train with full data
  - `python ml/train.py --dataset [Dataset_Name]`
- Train with splitted data
  - `python ml/train.py --dataset [Dataset_Name] --split`
- Example:
  - `python ml/train.py --dataset assistant`
  - `python ml/train.py --dataset assistant --split`

> `--split`, `--split True` and `--split False` are all accepted. Previously
> `--split False` was parsed as **True**, because `bool("False")` is true in Python.

### Test

- Parameter
  - `--dataset` [Name of the dataset, same as the csv file name]
  - `--split` [Default **False**]
    - if **True**, evaluate on the held-out test split
    - if **False**, evaluate on the full dataset
- Test with full data
  - `python ml/test.py --dataset [Dataset_Name]`
- Test with splitted data
  - `python ml/test.py --dataset [Dataset_Name] --split`
- Example:
  - `python ml/test.py --dataset assistant`

Reports are written to `ml/model_files/model_performance/<dataset>/`:

- `intent_report.json`, `intent_errors.json`, `intent_successes.json`
- `DIETClassifier_report.json` (entity extraction) and its errors/successes
- `intent_confusion_matrix.png`, `intent_histogram.png`
- `DIETClassifier_confusion_matrix.png`, `DIETClassifier_histogram.png`

> Rasa 3 names these plots itself. The Rasa 1.x code asked for `confmat.png`
> and `hist.png`; those options no longer exist.

### Inference

- Command line query testing
- Parameter
  - `--dataset` [Name of the dataset, same as the csv file name]
- `LOG_LEVEL=DEBUG` shows Rasa's full parse output for each message
- Inference file run
  - `python ml/inference.py --dataset [Dataset_Name]`
- Example:
  - `python ml/inference.py --dataset assistant`

### Predict

- Test a query list from a text file, with one query per line

- Parameter
  - `--dataset` [Name of the dataset, same as the csv file name]
  - `--inp` [Input file name]
  - `--out` [Output file name. A `.csv` extension produces a **csv** file, anything else a **text** file]
- Predict file run
  - `python ml/predict.py --dataset [Dataset_Name] --inp [Input_Filename] --out [Output_Filename]`
- Example:
  - Text file generate:
    - `python ml/predict.py --dataset assistant --inp query_list.txt --out predict_list.txt`
  - CSV file generate
    - `python ml/predict.py --dataset assistant --inp query_list.txt --out predict_list.csv`

### Backup & Restore

- Back up / restore a dataset's data, configuration, weights and reports.

- Parameter
  - `--dataset` [Name of the dataset, same as the csv file name]
  - `--foldername` [For restore: the backup folder name under `previous_version/`]
- Backup file run
  - `python ml/backup.py --dataset [Dataset_Name]`
- Restore file run
  - `python ml/backup.py --dataset [Dataset_Name] --foldername [Folder_Name]`
- Example:
  - Backup:
    - `python ml/backup.py --dataset assistant`
  - Restore
    - `python ml/backup.py --dataset assistant --foldername assistant_10_Nov_10_56_44_PM`

### Log Data

- `tensorboard --logdir ml/model_files/model_weight/tensorboard/[Dataset_Name]`
- Example
  - `tensorboard --logdir ml/model_files/model_weight/tensorboard/assistant`

## API

### Run

```bash
python main.py
```

### Configuration

All settings are environment variables; every one has a safe default.

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATASET` | `assistant` | Which trained model to serve |
| `CORS_ORIGINS` | *(unset)* | Comma-separated allowed origins. Unset means **no** cross-origin access |
| `CONFIDENCE_THRESHOLD` | `0.5` | Score at or above which `status` is `true` |
| `MAX_INPUT_LENGTH` | `1000` | Longest accepted message |
| `HOST` / `PORT` | `127.0.0.1` / `5000` | Dev-server bind address |
| `FLASK_DEBUG` | off | Enable the Werkzeug debugger (**never** in production) |
| `LOG_LEVEL` | `INFO` | Rasa/root log level. `DEBUG` shows every parse |

Example:

```bash
CORS_ORIGINS="https://app.example.com" PORT=8080 python main.py
```

### Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Service metadata |
| `GET` | `/health` | `200` when the model is loaded, `503` otherwise |
| `POST` | `/chat` | `{"user_input": "..."}` — preferred |
| `GET` | `/chat/<text>` | Kept for backwards compatibility |

```bash
curl -X POST http://127.0.0.1:5000/chat \
     -H 'Content-Type: application/json' \
     -d '{"user_input": "play a hindi song"}'
```

```json
{
  "sentence": "play a hindi song",
  "intent": "youtube",
  "confidence": 0.99,
  "threshold": 0.5,
  "status": true,
  "entities": { "key_name": ["hindi", "song"] }
}
```

`POST /chat` is preferred because query text sent in a URL path ends up in
access logs, proxy logs and browser history.

### Deployment

Flask's development server is single-process and not hardened. Run the app
behind a WSGI server:

```bash
pip install gunicorn
gunicorn --workers 1 --threads 4 --bind 0.0.0.0:8080 'main:app'
```

Use one worker per process: each one loads its own copy of the model.

## Security

Dependencies were moved to patched releases. Resolved advisories:

| Package | Was | Now | Advisories fixed |
| --- | --- | --- | --- |
| `rasa` | 1.10.14 | 3.6.21 | CVE-2024-49375 (critical, RCE via remote model loading), CVE-2021-41127 (high, arbitrary file write) |
| `Flask-Cors` | 3.0.9 | >= 6.0.5 | CVE-2024-6221, CVE-2024-1681, CVE-2024-6839, CVE-2024-6844, CVE-2024-6866 |
| `numpy` | < 1.19 | 1.23.x (pinned by Rasa) | CVE-2021-41495, CVE-2021-41496, CVE-2021-33430, CVE-2021-34141 |
| `spacy` | 2.3.2 | 3.8.x | — (kept current) |
| `pandas` | 1.1.2 | 2.x | — (kept current) |

`.github/dependabot.yml` keeps these under weekly review.

### Known residual risk — read this before closing the alerts

Upgrading Rasa removes the alerts on the packages this project names directly,
but it does **not** clear the digest. Rasa 3.6.21 is the end of the Apache-2.0
Rasa Open Source line (June 2023), and it hard-pins most of its own dependency
tree. A scan of the resolved environment (191 packages, OSV data) still reports
**63 advisories across 12 transitive packages**:

| Package | Pinned at | CVEs | Severity |
| --- | --- | --- | --- |
| `aiohttp` | 3.9.5 | 34 | 2 high, 17 moderate, 15 low |
| `keras` | 2.12.0 | 14 | 1 critical, 6 high, 6 moderate, 1 low |
| `skops` | 0.9.0 | 4 | 4 high |
| `protobuf` | 4.23.3 | 2 | 2 high |
| `setuptools` | 70.3.0 | 2 | 1 high, 1 moderate |
| `wheel` | 0.45.1 | 1 | 1 high |
| `scikit-learn` | 1.1.3 | 1 | 1 moderate |
| `pydantic` | 1.10.9 | 1 | 1 moderate |
| `pymongo` | 4.3.3 | 1 | 1 moderate |
| `dnspython` | 2.3.0 | 1 | 1 moderate |
| `Sanic-Cors` | 2.0.1 | 1 | 1 moderate |
| `sentry-sdk` | 1.14.0 | 1 | 1 low |

**None of these can be upgraded while Rasa is a dependency.** Rasa 3.6.21
requires `packaging>=20.0,<21.0`, which transitively blocks even a `wheel`
bump. Verified with `uv pip install --dry-run` for every package above — each
one resolves to "No solution found".

Most are also not reachable from this project's code path: `aiohttp`,
`Sanic-Cors`, `pymongo`, `dnspython` and `sentry-sdk` back Rasa's own server,
database connectors and telemetry, none of which this repo starts. The ones
worth understanding are `keras` and `skops`, whose advisories are largely about
**deserialising untrusted model files** — which matters only if you load a
model archive you did not train yourself. Train your own models and do not
load third-party `.tar.gz` archives.

To actually reach zero, the classifier would have to move off Rasa (for
example to a scikit-learn or sentence-transformers pipeline). That is a
rewrite, not an upgrade, and it would lose DIET's joint intent + entity
extraction. This upgrade was scoped to keep Rasa.

### Other hardening in this repo

- CORS is opt-in. The previous `CORS(app)` reflected every origin.
- The Flask debugger is off unless `FLASK_DEBUG` is set. It previously ran with
  `debug=True`, which exposes an interactive Python console on any traceback.
- `train.py` invokes the Rasa CLI via an argument list instead of `os.system`
  with an interpolated dataset name.
- `backup.py --foldername` rejects path separators.
- The vendored `installation/get-pip.py` (a base85 blob of pip 20.2.4) was
  removed; `python -m venv` provides pip already.
