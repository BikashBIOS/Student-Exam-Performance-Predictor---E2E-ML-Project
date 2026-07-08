# 🎓 Student Exam Performance Predictor — End-to-End ML Project

> A production-grade Machine Learning application that predicts a student's **Math score** based on demographic and academic features. Built as a complete E2E ML project covering EDA, model training with hyperparameter tuning, a Flask web app, and multi-cloud deployment on AWS (Elastic Beanstalk, EC2 + Docker + CI/CD) and Azure.

---

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [Project Structure](#project-structure)
- [ML Pipeline Architecture](#ml-pipeline-architecture)
- [Quick Start & Setup](#quick-start--setup)
- [requirements.txt & setup.py — Explained](#requirementstxt--setuppy--explained)
- [Core Infrastructure: `src/exception.py`](#core-infrastructure-srcexceptionpy)
- [Core Infrastructure: `src/logger.py`](#core-infrastructure-srcloggerpy)
- [Core Infrastructure: `src/utils.py`](#core-infrastructure-srcutilspy)
- [Component 1: `src/components/data_ingestion.py`](#component-1-srccomponentsdata_ingestionpy)
- [Component 2: `src/components/data_transformation.py`](#component-2-srccomponentsdata_transformationpy)
- [Component 3: `src/components/model_trainer.py`](#component-3-srccomponentsmodel_trainerpy)
- [Prediction Pipeline: `src/pipeline/predict_pipeline.py`](#prediction-pipeline-srcpipelinepredict_pipelinepy)
- [Flask Web Application: `app.py` & `application.py`](#flask-web-application-apppy--applicationpy)
- [HTML Templates: `templates/home.html`](#html-templates-templateshomehtml)
- [Jupyter Notebooks: EDA & Model Training](#jupyter-notebooks-eda--model-training)
- [Deployment Guide](#deployment-guide)
  - [AWS Elastic Beanstalk](#aws-elastic-beanstalk-deployment)
  - [AWS EC2 + Docker + CI/CD Pipeline](#aws-ec2--docker--cicd-pipeline)
  - [Azure Container Registry + Web App](#azure-container-registry--web-app)
- [Dockerfile Explained](#dockerfile-explained)
- [Data Flow: End-to-End Architecture](#data-flow-end-to-end-architecture)
- [Results](#results)

---

## What This Project Does

This project answers a simple question: **can we predict a student's math score based on their background?**

The dataset contains the following features for each student:

| Feature | Type | Values |
|---|---|---|
| `gender` | Categorical | male / female |
| `race_ethnicity` | Categorical | group A / B / C / D / E |
| `parental_level_of_education` | Categorical | high school / some college / bachelor's / master's / associate's / some high school |
| `lunch` | Categorical | standard / free/reduced |
| `test_preparation_course` | Categorical | none / completed |
| `reading_score` | Numerical | 0–100 |
| `writing_score` | Numerical | 0–100 |
| **`math_score`** | **Target** | **0–100** |

The end result is a live web application where anyone can fill in these details and get a predicted math score.

---

## Project Structure

```
mlprojects/
│
├── src/                          # All reusable Python source code (a Python package)
│   ├── __init__.py               # Makes src/ a Python package
│   ├── exception.py              # Custom exception class with file + line info
│   ├── logger.py                 # Timestamped file logging setup
│   ├── utils.py                  # Shared utilities: save/load pickle, model evaluation
│   │
│   ├── components/               # The three ML pipeline stages
│   │   ├── __init__.py
│   │   ├── data_ingestion.py     # Stage 1: Load raw data, train/test split
│   │   ├── data_transformation.py # Stage 2: Impute, encode, scale → preprocessor.pkl
│   │   └── model_trainer.py      # Stage 3: Train 8 models, hyperparameter tune, save best → model.pkl
│   │
│   └── pipeline/                 # Runtime pipelines
│       ├── __init__.py
│       ├── predict_pipeline.py   # Load pkl files, accept web input, return prediction
│       └── train_pipeline.py     # (Orchestrator placeholder)
│
├── notebook/                     # Jupyter notebooks for EDA and model prototyping
│   └── data/
│       └── stud.csv              # Raw dataset
│
├── artifacts/                    # Auto-generated output files (gitignored)
│   ├── data.csv                  # Raw data copy
│   ├── train.csv                 # Training split (80%)
│   ├── test.csv                  # Test split (20%)
│   ├── preprocessor.pkl          # Saved sklearn ColumnTransformer
│   └── model.pkl                 # Saved best ML model
│
├── templates/                    # Flask HTML templates
│   ├── index.html                # Landing page
│   └── home.html                 # Prediction form + results display
│
├── .ebextensions/
│   └── python.config             # AWS Elastic Beanstalk WSGI configuration
│
├── app.py                        # Flask app entry point (for local/Docker/EC2)
├── application.py                # Flask app entry point (renamed for AWS EB)
├── setup.py                      # Python package installer
├── requirements.txt              # All Python dependencies
└── Dockerfile                    # Container definition for Docker deployment
```

---

## ML Pipeline Architecture

The training pipeline runs in three sequential stages, all triggered by executing `data_ingestion.py`:

```
┌──────────────────────────────────────────────────────────────────────┐
│  python src/components/data_ingestion.py                             │
│                                                                      │
│  Stage 1: DataIngestion                                              │
│  ┌─────────────────────────────────┐                                 │
│  │  notebook/data/stud.csv         │ ← raw CSV                      │
│  │         ↓                       │                                 │
│  │  artifacts/data.csv             │ ← full raw copy                 │
│  │  artifacts/train.csv (80%)      │ ← training split                │
│  │  artifacts/test.csv  (20%)      │ ← test split                    │
│  └─────────────────────────────────┘                                 │
│                ↓                                                     │
│  Stage 2: DataTransformation                                         │
│  ┌─────────────────────────────────┐                                 │
│  │  Numerical: [writing, reading]  │                                 │
│  │    → SimpleImputer(median)      │                                 │
│  │    → StandardScaler             │                                 │
│  │  Categorical: [gender, race...] │                                 │
│  │    → SimpleImputer(most_freq)   │                                 │
│  │    → OneHotEncoder              │                                 │
│  │  ColumnTransformer combines ↑   │                                 │
│  │  artifacts/preprocessor.pkl     │ ← saved for inference           │
│  └─────────────────────────────────┘                                 │
│                ↓                                                     │
│  Stage 3: ModelTrainer                                               │
│  ┌─────────────────────────────────┐                                 │
│  │  8 Models × GridSearchCV tuning │                                 │
│  │  → evaluate_models() → R² dict  │                                 │
│  │  → best model selected          │                                 │
│  │  artifacts/model.pkl            │ ← saved best model              │
│  └─────────────────────────────────┘                                 │
└──────────────────────────────────────────────────────────────────────┘

Inference (web app):
User Form → CustomData → DataFrame → preprocessor.pkl → model.pkl → Math Score
```

---

## Quick Start & Setup

### Step 1 — Create GitHub Repo & Local Environment

```bash
# 1. Create your project folder
mkdir mlprojects && cd mlprojects

# 2. Open VS Code in the folder (optional)
code .

# 3. Create and activate Python virtual environment
python -m venv venv
source venv/bin/activate       # Linux/Mac
# venv\Scripts\activate        # Windows

# 4. Initialize Git and push
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/mlprojects.git
git push -u origin main
```

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

This also installs the `src/` folder as a local package (due to `-e .` in requirements.txt).

### Step 3 — Run the Training Pipeline

```bash
python src/components/data_ingestion.py
```

This runs all three stages (ingestion → transformation → training) and produces `artifacts/model.pkl` and `artifacts/preprocessor.pkl`.

### Step 4 — Run the Web App

```bash
python app.py
# Visit: http://localhost:8080
```

---

## requirements.txt & setup.py — Explained

### requirements.txt

```
pandas          # Data loading, manipulation, DataFrames
numpy           # Numerical arrays, np.c_ column concatenation
seaborn         # Statistical visualisation (used in EDA notebooks)
matplotlib      # Base plotting library (used in EDA notebooks)
scikit-learn    # Core ML: pipelines, preprocessing, models, metrics
xgboost         # Extreme Gradient Boosting regressor
catboost        # Yandex's gradient boosting — handles categoricals natively
dill            # Extended pickle: serialises Python objects (used in save_object)
flask           # Web framework for the prediction API
-e .            # Install the src/ folder itself as an editable local package
```

**Why `dill` instead of just `pickle`?** Regular `pickle` can't serialise certain Python objects like lambda functions or nested classes. `dill` extends pickle and handles them safely.

**What does `-e .` do?** It tells pip to install the current directory as an editable package using `setup.py`. This is what allows every file in `src/` to be imported as `from src.exception import CustomException` anywhere in the project — without hacking `sys.path`.

---

### setup.py

```python
from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name='mlproject',
    version='0.0.1',
    author='Bikash',
    author_email='bikashojha101@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)
```

**Line-by-line:**

`HYPEN_E_DOT = '-e .'` — A constant for the self-install directive that appears in requirements.txt. It's defined as a constant so it can be reliably detected and removed.

`get_requirements(file_path)` — Reads requirements.txt line by line, strips the trailing newline `\n` from each line, and removes the `-e .` entry before returning. The `-e .` is a pip directive that has no meaning inside `install_requires=` — if left in, `setup()` would crash. This function cleanly bridges between the two formats.

`find_packages()` — Automatically scans the project directory and finds all folders containing an `__init__.py` file, registering them as importable Python packages. This is what makes `from src.exception import CustomException` work — `src/` has an `__init__.py` so `find_packages()` includes it.

`install_requires=get_requirements('requirements.txt')` — Passes the cleaned dependency list to `setup()`, so `pip install -e .` installs everything in requirements.txt as dependencies of this package.

---

## Core Infrastructure: `src/exception.py`

Every error in the ML pipeline is wrapped through this module, so failures always tell you **which file**, **which line**, and **what went wrong** — rather than a bare Python traceback.

```python
import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in Python script name [{0}] line number [{1}] error message[{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
```

**`error_message_detail(error, error_detail: sys)`**

`error_detail.exc_info()` — Calls `sys.exc_info()`, which returns a tuple of `(type, value, traceback)`. The `_,_,exc_tb` pattern unpacks and discards the first two, keeping only the traceback object.

`exc_tb.tb_frame.f_code.co_filename` — Navigates the traceback object to get the exact filename where the error occurred: `.tb_frame` gets the current stack frame, `.f_code` gets the code object, `.co_filename` gets the source file path.

`exc_tb.tb_lineno` — Gets the exact line number within that file where the exception was raised.

The formatted message looks like: `"Error occurred in Python script name [src/components/data_ingestion.py] line number [35] error message[FileNotFoundError]"`

**`class CustomException(Exception)`**

`super().__init__(error_message)` — Calls the parent `Exception` class constructor with the original error message, so this class behaves as a proper Python exception.

`self.error_message = error_message_detail(...)` — Immediately calls the detail function to build the rich error string, stored as an instance attribute.

`__str__` — Overrides the default string representation so when `raise CustomException(e, sys)` is caught and printed, it shows the rich message instead of the bare original error.

**Usage throughout the project:**
```python
try:
    # some risky code
except Exception as e:
    raise CustomException(e, sys)
```

---

## Core Infrastructure: `src/logger.py`

Provides timestamped file logging so every step of the pipeline is recorded permanently, even after the process ends.

```python
import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
```

**`LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"`**

Creates a unique log filename based on the exact timestamp the program started — e.g. `06_21_2025_14_30_00.log`. Every pipeline run gets its own log file, so nothing gets overwritten.

**`logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)`**

Builds the path `<project_root>/logs/06_21_2025_14_30_00.log/`. Note that `LOG_FILE` (the filename) is being used as a directory name here — a quirk in this code, but `os.makedirs` creates it without issue.

**`os.makedirs(logs_path, exist_ok=True)`**

Creates the `logs/` directory (and subdirectory) if they don't exist. `exist_ok=True` prevents crashes if the directory already exists.

**`logging.basicConfig(...)`**

Configures Python's built-in logging module:
- `filename=LOG_FILE_PATH` — write logs to file instead of console
- `format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"` — log format: `[timestamp] line_number logger_name - INFO - your message`
- `level=logging.INFO` — only record INFO and above (not DEBUG). Logs look like: `[ 2025-06-21 14:30:01,234 ] 45 root - INFO - Read the dataset as dataframe`

**Usage throughout the project:**
```python
logging.info("Read the dataset as data frame")
logging.info("Train test split initiated")
```

---

## Core Infrastructure: `src/utils.py`

Shared utility functions used by all three pipeline components.

```python
import os, sys, numpy as np, pandas as pd, dill, pickle
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
```

---

### `save_object(file_path, obj)`

**Purpose:** Serialise any Python object (a trained sklearn model, a preprocessing pipeline etc.) to disk as a binary `.pkl` file so it can be reloaded later without retraining.

`os.path.dirname(file_path)` — Extracts the directory path from the full file path. If `file_path = "artifacts/model.pkl"`, this gives `"artifacts"`.

`os.makedirs(dir_path, exist_ok=True)` — Creates the `artifacts/` directory if it doesn't exist. `exist_ok=True` is safe — no crash if the directory already exists.

`with open(file_path, "wb")` — Opens the file in **write binary** mode (`"wb"`). Pickle files are binary, not text.

`dill.dump(obj, file_obj)` — Serialises the object. `dill` is used instead of `pickle` because it handles more complex Python objects like sklearn `Pipeline` objects with nested custom transformers.

---

### `evaluate_models(X_train, y_train, X_test, y_test, models, params)`

**Purpose:** Run GridSearchCV hyperparameter tuning for every model in the `models` dict, evaluate R² score on the test set, and return a report dictionary.

```python
for i in range(len(list(models))):
    model = list(models.values())[i]      # e.g. LinearRegression()
    para = params[list(models.keys())[i]] # e.g. {} or {'n_estimators': [8,16,...]}
```

Iterates through all models by index. Each model is paired with its hyperparameter grid from the `params` dict.

```python
gs = GridSearchCV(model, para, cv=3)
gs.fit(X_train, y_train)
```

`GridSearchCV` exhaustively tries every combination of hyperparameter values using 3-fold cross-validation (`cv=3`). For each combination, it splits `X_train` into 3 parts, trains on 2, validates on 1, and repeats 3 times. The combination with the best average validation score becomes `gs.best_params_`.

```python
model.set_params(**gs.best_params_)
model.fit(X_train, y_train)
```

`set_params(**gs.best_params_)` — Applies the best hyperparameters found by GridSearchCV back onto the original model object. `**` unpacks the dict into keyword arguments, e.g. `model.set_params(n_estimators=256, learning_rate=0.1)`.

Then `model.fit(X_train, y_train)` trains the model on the full training set using those best parameters (not just the CV folds).

```python
train_model_score = r2_score(y_train, y_train_pred)
test_model_score = r2_score(y_test, y_test_pred)
report[list(models.keys())[i]] = test_model_score
```

**R² (R-squared / Coefficient of Determination):** A metric from 0 to 1 (higher is better) that measures how much of the variance in the target variable the model explains. R²=1.0 means perfect prediction; R²=0.0 means the model does no better than always predicting the mean.

Only the **test** score is stored in `report` — training score is computed but not used in selection (to avoid rewarding overfitted models).

---

### `load_object(file_path)`

```python
with open(file_path, "rb") as file_obj:
    return pickle.load(file_obj)
```

Opens the `.pkl` file in **read binary** mode and deserialises it back into the original Python object (model or preprocessor). Note: `pickle` is used here (not `dill`), because the objects were saved with `dill` — `pickle.load` can read `dill`-saved files.

---

## Component 1: `src/components/data_ingestion.py`

The entry point of the entire training pipeline. Reads raw data, makes a copy in `artifacts/`, and splits it into train and test sets.

```python
import os, sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformationConfig, DataTransformation
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")
```

**`@dataclass`** — Python's `dataclass` decorator auto-generates `__init__`, `__repr__` and other dunder methods for a class that primarily holds data. Instead of writing a full `__init__` constructor, you declare attributes with type annotations. Here it acts as a **configuration holder** — a single place that defines all output file paths for this stage, making paths easy to change without hunting through method code.

The three paths defined:
- `train_data_path` → where the 80% training set CSV will be saved
- `test_data_path` → where the 20% test set CSV will be saved
- `raw_data_path` → where the full raw dataset copy will be saved

All go into the `artifacts/` folder, which is auto-created.

```python
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv(r'notebook\data\stud.csv')
            logging.info('Read the dataset as data frame')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train test split initiated")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion of Data completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
```

`self.ingestion_config = DataIngestionConfig()` — Instantiates the config dataclass. Now all file paths are accessible as `self.ingestion_config.train_data_path` etc.

`df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)` — Saves the full raw DataFrame as a CSV. `index=False` prevents pandas from adding an auto-generated row index column. `header=True` keeps the column names.

`train_test_split(df, test_size=0.2, random_state=42)` — Splits the DataFrame into 80% training and 20% test. `random_state=42` seeds the random number generator so the split is identical on every run — critical for reproducibility.

The method returns a tuple of the two CSV paths, which are passed directly to the next stage.

```python
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))
```

This `if __name__ == "__main__"` block is the **full pipeline orchestrator**. Running `python src/components/data_ingestion.py` triggers all three stages in sequence and prints the final R² score.

---

## Component 2: `src/components/data_transformation.py`

Transforms raw CSV data into normalised, encoded numpy arrays ready for ML models. Produces and saves `artifacts/preprocessor.pkl`.

```python
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
```

Single config field: where to save the trained preprocessor object.

### `get_data_transformer_object()` — The Preprocessing Pipeline

```python
numerical_columns = ["writing_score", "reading_score"]
categorical_columns = [
    "gender", "race_ethnicity", "parental_level_of_education",
    "lunch", "test_preparation_course",
]

num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("one_hot_encoder", OneHotEncoder())
])

preprocessor = ColumnTransformer([
    ("num_pipeline", num_pipeline, numerical_columns),
    ("cat_pipeline", cat_pipeline, categorical_columns)
])

return preprocessor
```

**Numerical Pipeline:**

`SimpleImputer(strategy="median")` — Fills any missing values in numerical columns with the column median. Median is preferred over mean for skewed data because it's resistant to outliers. If a student's `reading_score` is missing, it gets replaced with the median reading score of all students.

`StandardScaler()` — Standardises each numerical feature to have mean=0 and standard deviation=1, using the formula: `z = (x - mean) / std`. This prevents features with larger scales (e.g. a score of 95) from dominating features with smaller scales in distance-based models.

**Categorical Pipeline:**

`SimpleImputer(strategy="most_frequent")` — Fills missing categorical values with the most common category (mode). If `gender` is missing, it's filled with whichever value appears most often.

`OneHotEncoder()` — Converts categorical strings into binary columns. For example, `gender = ["male", "female"]` becomes two columns: `gender_male` (1 if male, else 0) and `gender_female`. This is required because ML algorithms work on numbers, not strings.

**`ColumnTransformer`:**

Applies the `num_pipeline` only to `numerical_columns` and the `cat_pipeline` only to `categorical_columns`, then horizontally concatenates the results. This is the correct way to apply different preprocessing steps to different column types in sklearn.

---

### `initiate_data_transformation()` — Fit and Transform

```python
input_feature_train_df = train_df.drop(columns=[target_column_name])   # X_train
target_feature_train_df = train_df[target_column_name]                  # y_train

input_feature_test_df = test_df.drop(columns=[target_column_name])     # X_test
target_feature_test_df = test_df[target_column_name]                   # y_test
```

Separates features (X) from target (y = `math_score`).

```python
input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
```

**Critical distinction:**
- `.fit_transform()` on training data — Learns the statistics (mean, std, medians, category lists) FROM the training data, then applies the transformation. The preprocessor "memorises" these statistics.
- `.transform()` on test data — Applies the SAME statistics learned from training. Never fits on test data — that would cause **data leakage**, where the model indirectly sees test information during training.

```python
train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
```

`np.c_[...]` — NumPy's column-wise concatenation operator. Attaches the target column (`math_score`) back onto the right side of the transformed feature matrix. Result: each row is `[feature1, feature2, ..., featureN, math_score]`.

```python
save_object(
    file_path=self.data_transformation_config.preprocessor_obj_file_path,
    obj=preprocessing_obj
)
```

Saves the **fitted** preprocessor (with all learned statistics) to `artifacts/preprocessor.pkl`. This exact object is reloaded during inference so web app predictions use the same scaling parameters as training.

---

## Component 3: `src/components/model_trainer.py`

Trains 8 ML models with hyperparameter tuning, selects the best by R² score, and saves it to `artifacts/model.pkl`.

```python
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
```

### The 8 Models

```python
models = {
    "Linear Regression":        LinearRegression(),
    "K-Neighbors Regressor":    KNeighborsRegressor(),
    "Decision Tree":            DecisionTreeRegressor(),
    "Random Forest Regressor":  RandomForestRegressor(),
    "XGBRegressor":             XGBRegressor(),
    "CatBoosting Regressor":    CatBoostRegressor(verbose=False),
    "AdaBoost Regressor":       AdaBoostRegressor(),
    "Gradient Boosting":        GradientBoostingRegressor()
}
```

| Model | Type | Strength |
|---|---|---|
| **Linear Regression** | Linear | Fast baseline, interpretable |
| **K-Neighbors** | Instance-based | Non-parametric, works on local patterns |
| **Decision Tree** | Tree | Interpretable, handles non-linearity |
| **Random Forest** | Ensemble (Bagging) | Reduces variance, robust to noise |
| **XGBoost** | Ensemble (Boosting) | High accuracy, handles missing data |
| **CatBoost** | Ensemble (Boosting) | Handles categoricals, fast convergence |
| **AdaBoost** | Ensemble (Boosting) | Focuses on hard examples |
| **Gradient Boosting** | Ensemble (Boosting) | Flexible loss functions |

### Hyperparameter Grids

```python
params = {
    "Decision Tree": {
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    },
    "K-Neighbors Regressor": {
        'n_neighbors': [5, 7, 9, 11],
    },
    "Random Forest Regressor": {
        'n_estimators': [8, 16, 32, 64, 128, 256]
    },
    "Gradient Boosting": {
        'learning_rate': [.1, .01, .05, .001],
        'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
        'n_estimators': [8, 16, 32, 64, 128, 256]
    },
    "Linear Regression": {},          # No hyperparameters to tune
    "XGBRegressor": {
        'learning_rate': [.1, .01, .05, .001],
        'n_estimators': [8, 16, 32, 64, 128, 256]
    },
    "CatBoosting Regressor": {
        'depth': [6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'iterations': [30, 50, 100]
    },
    "AdaBoost Regressor": {
        'learning_rate': [.1, .01, 0.5, .001],
        'n_estimators': [8, 16, 32, 64, 128, 256]
    }
}
```

**Key hyperparameter concepts:**

`n_estimators` — Number of trees in an ensemble. More trees generally mean better accuracy but more training time. GridSearchCV picks the optimal count.

`learning_rate` — How much each new tree corrects the errors of previous trees (in boosting models). Smaller = more conservative, requires more trees. Higher = faster but can overshoot.

`subsample` — Fraction of training samples used per tree (Gradient Boosting). Values below 1.0 introduce randomness to reduce overfitting.

`depth` / `criterion` — Tree structure controls. Depth limits how many levels a tree can grow; criterion defines what "quality" of a split means.

### Selecting the Best Model

```python
model_report: dict = evaluate_models(
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test,
    models=models, params=params
)

best_model_score = max(sorted(model_report.values()))
best_model_name = list(model_report.keys())[
    list(model_report.values()).index(best_model_score)
]
best_model = models[best_model_name]

if best_model_score < 0.6:
    raise CustomException("No best model found")
```

`max(sorted(model_report.values()))` — Sorts all R² scores and picks the highest. `sorted()` ensures deterministic order before `max()`.

`list(model_report.values()).index(best_model_score)` — Finds the position of the best score in the values list, then uses that index to get the corresponding model name from the keys list. This maps score → model name.

`if best_model_score < 0.6` — A quality gate. If no model achieves at least 60% R², something is wrong (bad data, wrong features etc.) and a custom exception is raised instead of saving a poor model.

```python
save_object(
    file_path=self.model_trainer_config.trained_model_file_path,
    obj=best_model
)

predicted = best_model.predict(X_test)
r2_square = r2_score(y_test, predicted)
return r2_square
```

Saves the winning model and returns the final R² score, which gets printed by the orchestrator in `data_ingestion.py`.

---

## Prediction Pipeline: `src/pipeline/predict_pipeline.py`

Used by the Flask web app at runtime to accept form inputs and return a predicted math score.

```python
class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)
```

`load_object(file_path=model_path)` — Loads the trained best model from `artifacts/model.pkl`. This is the model selected and saved by `model_trainer.py` during training.

`load_object(file_path=preprocessor_path)` — Loads the fitted `ColumnTransformer` from `artifacts/preprocessor.pkl`. This contains all the learned statistics (medians, means, category encodings) from training.

`preprocessor.transform(features)` — Applies ONLY `.transform()` (not `fit_transform`). The preprocessor already knows all statistics — it just applies them to the new input data. This ensures the web app prediction uses identical scaling to what the model was trained on.

`model.predict(data_scaled)` — Passes the scaled feature array to the model and gets back a numpy array of predictions. Since there's only one input row from the web form, the result is a 1-element array.

---

### `class CustomData` — Web Form → DataFrame

```python
class CustomData:
    def __init__(self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
```

`CustomData` is a **data transfer object** (DTO) that bridges the HTML form and the ML pipeline. It takes the 7 raw form fields as constructor arguments, stores them as instance attributes, and then `get_data_as_data_frame()` converts them to a properly structured pandas DataFrame.

**Why a DataFrame?** The sklearn `ColumnTransformer` preprocessor was trained on a DataFrame with specific column names. Passing it a dict or list would cause column-name mismatch errors. Each value is wrapped in a list `[self.gender]` because DataFrames expect iterable values — a single string would create a length-1 series instead of a single-row column.

---

## Flask Web Application: `app.py` & `application.py`

These two files are **identical in logic** — the only difference is how the Flask app is launched:

- `app.py` — runs on `host="0.0.0.0", port=8080` (used for local dev, Docker, EC2)
- `application.py` — runs on `host="0.0.0.0"` with no port (AWS Elastic Beanstalk default port 5000, handled by EB's WSGI config)

```python
import pickle
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application
```

`application = Flask(__name__)` — Creates the Flask application instance. `__name__` tells Flask where to find templates and static files (relative to this script's location). The `application` variable name is **required** by AWS Elastic Beanstalk's WSGI convention — EB looks for a callable named `application`.

`app = application` — Creates a second reference to the same object. This allows you to use `@app.route()` decorators (which is more conventional Flask style) while keeping the `application` name available for WSGI.

```python
@app.route('/')
def index():
    return render_template('index.html')
```

The root route `/` serves the landing page from `templates/index.html`.

```python
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
```

`methods=['GET', 'POST']` — This route handles two HTTP methods. `GET` (when the user navigates to `/predictdata` in their browser) shows the empty form. `POST` (when the user submits the form) runs the prediction.

`request.form.get('gender')` — Reads the value of the HTML `<select>` or `<input>` element with `name="gender"` from the submitted form data.

`float(request.form.get('writing_score'))` — Converts the string form input to a float. HTML form data is always strings; the ML model expects numbers.

`results[0]` — `predict_pipeline.predict()` returns a numpy array. Since there's only one form submission (one student), `results[0]` extracts the single predicted math score value to pass back to the template.

`return render_template('home.html', results=results[0])` — Re-renders the prediction form page, this time with the `results` variable injected into the Jinja2 template context. The template displays it in `{{ results }}`.

`app.run(host="0.0.0.0", port=8080)` — `host="0.0.0.0"` binds to all network interfaces (not just localhost), which is required when running in Docker or on a remote server like EC2.

---

## HTML Templates: `templates/home.html`

The prediction form collects all 7 input features from the user.

```html
<form action="{{ url_for('predict_datapoint') }}" method="post">
```

`{{ url_for('predict_datapoint') }}` — Jinja2 template syntax. Flask's `url_for()` dynamically generates the URL for the `predict_datapoint` Python function (which maps to `/predictdata`). This is safer than hardcoding `/predictdata` because it auto-updates if the route ever changes.

`method="post"` — The form submits data via HTTP POST, which sends the form values in the request body (not visible in the URL). This is important for form submissions that modify state or send sensitive data.

```html
<select class="form-control" name="gender" required>
    <option value="male">Male</option>
    <option value="female">Female</option>
</select>
```

Each `<select>` dropdown's `name` attribute must exactly match what `request.form.get('...')` looks for in `app.py`. The `value` attributes correspond to the actual category strings in the training dataset.

```html
<input type="number" name="reading_score" min='0' max='100' />
```

`type="number"` — HTML5 input that only accepts numeric values. `min` and `max` constrain the range to valid score values on the client side.

```html
<h2>THE prediction is {{ results }}</h2>
```

Jinja2 renders the `results` variable passed from Flask. When the page is first loaded (GET request), `results` is `None` and nothing shows. After form submission (POST), it displays the predicted math score.

---

## Jupyter Notebooks: EDA & Model Training

Two notebooks in the `notebook/` folder serve as the **research and prototyping** phase before the production code was written:

### EDA Notebook (`EDA.ipynb`)

The Exploratory Data Analysis notebook covers:

- **Data Loading:** `pd.read_csv('data/stud.csv')` to load the raw dataset
- **Shape & Info:** `df.shape`, `df.info()`, `df.describe()` to understand dimensions, dtypes, and basic statistics
- **Null Check:** `df.isnull().sum()` — confirms whether missing value imputation is actually needed
- **Duplicate Check:** `df.duplicated().sum()`
- **Univariate Analysis:** Distribution plots for each feature using `seaborn.histplot()`, `sns.countplot()`
- **Bivariate Analysis:** Correlation heatmaps (`sns.heatmap(df.corr())`), box plots of scores by gender/lunch/race to discover which features most influence math score
- **Key Findings:** e.g. students who completed the test preparation course score higher; standard lunch type correlates with better scores

### Model Training Notebook (`model_trainer.ipynb`)

The model prototyping notebook covers:

- **Train/Test Split:** `train_test_split(X, y, test_size=0.2, random_state=42)`
- **Encoding:** `OneHotEncoder` for categoricals, `StandardScaler` for numericals
- **`evaluate_model()` function:** Computes MSE, MAE, and R² for each model
- **All 8 Models Applied:** Each model is trained and scored; results compared in a table
- **Best Model Selection:** R² scores ranked; highest selected
- **Actual vs. Predicted Plot:** `plt.scatter(y_test, y_pred)` with a best-fit line drawn using `np.polyfit()` to visually validate the model's performance

These notebooks are exploratory — the code is then productionised into the `src/components/` files with proper logging, exception handling, and artifact saving.

---

## Deployment Guide

### AWS Elastic Beanstalk Deployment

Elastic Beanstalk manages the underlying infrastructure (EC2, load balancers, scaling groups) automatically. You only need to provide the application code.

#### 1. Configure WSGI — `.ebextensions/python.config`

```yaml
option_settings:
  "aws:elasticbeanstalk:container:python":
    WSGIPath: application:application
```

`WSGIPath: application:application` — Tells Elastic Beanstalk where to find the Flask app. Format: `module_name:wsgi_callable`. The first `application` is the Python filename (`application.py`) and the second `application` is the Flask app object (`application = Flask(__name__)`) inside that file. This is why `app.py` was renamed to `application.py`.

#### 2. Steps

```
1. Rename app.py → application.py (already done in repo)
2. Go to AWS Console → Elastic Beanstalk → Create Application
3. Select Python platform
4. Upload your code or connect to CodePipeline
5. Create Application → EB provisions the server automatically
6. Go to AWS CodePipeline → Create Pipeline
7. Source: GitHub repository → connect and select branch
8. Deploy: Elastic Beanstalk → select your app and environment
9. Create Pipeline → automatic deployments on every git push
```

---

### AWS EC2 + Docker + CI/CD Pipeline

A more advanced deployment where you control the server and set up a full CI/CD pipeline via GitHub Actions.

#### Dockerfile

```dockerfile
FROM python:3.14.0-bookworm
WORKDIR /app
COPY . /app

RUN apt update -y && apt install awscli -y

RUN pip install -r requirements.txt

CMD ["python3", "app.py"]
```

`FROM python:3.14.0-bookworm` — Base image: Debian "Bookworm" with Python 3.14 pre-installed. "Bookworm" is Debian 12's codename — a stable, production-appropriate base.

`WORKDIR /app` — Sets the working directory inside the container. All subsequent commands and file paths are relative to `/app`.

`COPY . /app` — Copies the entire project (all files from the current directory on your machine) into `/app` inside the container.

`RUN apt update -y && apt install awscli -y` — Installs the AWS CLI inside the container. Used for ECR (Elastic Container Registry) authentication and potential S3 operations.

`RUN pip install -r requirements.txt` — Installs all Python dependencies inside the container's isolated Python environment.

`CMD ["python3", "app.py"]` — The default command that runs when the container starts. Starts the Flask web server.

#### Full EC2 + Docker + GitHub Actions CI/CD Setup

```
Step 1 — Build & Test Docker locally:
  docker build -t student-performance-app .
  docker run -p 5000:5000 student-performance-app
  # Verify at http://localhost:5000

Step 2 — Create GitHub Workflow (.github/workflows/main.yaml):
  # Copy the "Deploy to Amazon ECS" template from GitHub Actions

Step 3 — Create AWS IAM User:
  - Permissions: AmazonEC2ContainerRegistryFullAccess, AmazonEC2FullAccess
  - Create Access Key → Download credentials CSV

Step 4 — Create ECR Repository:
  AWS Console → ECR → Create Repository
  Name: student-performance
  Note the repository URI (e.g. 123456789.dkr.ecr.us-east-1.amazonaws.com/student-performance)

Step 5 — Create EC2 Instance:
  - Name: student-performance
  - AMI: Ubuntu 64-bit
  - Instance type: t2.micro (free tier)
  - Enable: Allow HTTP, Allow HTTPS
  - Launch Instance

Step 6 — Install Docker on EC2 (via EC2 Instance Connect CLI):
  sudo apt-get update -y
  sudo apt-get upgrade
  curl -fsSL https://get.docker.com -o get-docker.sh
  sudo sh get-docker.sh
  sudo usermod -aG docker ubuntu
  newgrp docker

Step 7 — Configure GitHub Self-Hosted Runner on EC2:
  GitHub repo → Settings → Actions → Runners → New self-hosted runner
  Select Linux → Copy all the commands shown
  Paste and run in EC2 CLI
  Set runner name: self-hosted
  Run ./run.sh → runner shows green (idle) in GitHub

Step 8 — Add GitHub Secrets:
  AWS_ACCESS_KEY_ID         → from downloaded credentials
  AWS_SECRET_ACCESS_KEY     → from downloaded credentials
  AWS_REGION                → e.g. us-east-1
  AWS_ECR_LOGIN_URI         → ECR repository URI
  ECR_REPOSITORY_NAME       → student-performance

Step 9 — Push Code:
  Any push to main branch → GitHub Actions triggers →
  Builds Docker image → Pushes to ECR → Pulls to EC2 → Runs container

Step 10 — Expose Port on EC2:
  EC2 → Security Groups → Edit Inbound Rules
  Add: Custom TCP, Port 8080
  Visit: http://<EC2-PUBLIC-IP>:8080

⚠️ IMPORTANT: Delete all resources after testing to avoid AWS charges:
  - Terminate EC2 instance
  - Delete ECR repository
  - Delete IAM user
  - Remove GitHub Runner
```

---

### Azure Container Registry + Web App

An alternative cloud deployment using Microsoft Azure's managed container services.

```
Step 1 — Create Azure Container Registry:
  Azure Portal → Container Registry → Create
  Resource group: (your group)
  Registry name: (your registry name)
  After creating → Access Keys → Enable Admin user
  Note: Login server, Username, Password

Step 2 — Build and Push Docker Image:
  docker build -t <login-server>/application:latest .
  docker login <login-server>   (enter username + password from Step 1)
  docker push <login-server>/application:latest

Step 3 — Create Azure Web App:
  Azure Portal → Web App → Create
  Resource group: (same as Step 1)
  Runtime: Docker Container
  OS: Linux
  Docker tab → Single Container → select your registry and image

Step 4 — Enable Continuous Deployment:
  Web App → Deployment Center
  Source: GitHub Actions
  Organization: your GitHub username
  Repository: mlprojects
  Branch: main
  Save

Step 5 — GitHub Actions Auto-Created:
  .github/workflows/ will be auto-generated by Azure
  Every push to main → build → push to Container Registry → redeploy Web App
```

---

## Data Flow: End-to-End Architecture

```
TRAINING TIME
──────────────
notebook/data/stud.csv
         │
         ▼
 DataIngestion.initiate_data_ingestion()
         │  ├── artifacts/data.csv  (full raw)
         │  ├── artifacts/train.csv (80%)
         │  └── artifacts/test.csv  (20%)
         ▼
 DataTransformation.initiate_data_transformation()
         │  Numerical: SimpleImputer(median) → StandardScaler
         │  Categorical: SimpleImputer(most_freq) → OneHotEncoder
         │  ColumnTransformer.fit_transform(X_train)
         │  ColumnTransformer.transform(X_test)
         │  np.c_[features, math_score] → train_arr, test_arr
         │  ├── artifacts/preprocessor.pkl  ← fitted ColumnTransformer
         ▼
 ModelTrainer.initiate_model_trainer()
         │  GridSearchCV on 8 models
         │  Select best by R² score
         │  Quality gate: R² > 0.6
         │  └── artifacts/model.pkl  ← best trained model

INFERENCE TIME
──────────────
User fills HTML form (home.html)
         │  POST /predictdata
         ▼
 app.py: request.form.get(...)
         │
         ▼
 CustomData(gender, race, education, lunch, prep, reading, writing)
         │
         ▼
 .get_data_as_data_frame() → pandas DataFrame (1 row, 7 cols)
         │
         ▼
 PredictPipeline.predict(df)
         │  load_object(preprocessor.pkl).transform(df)
         │  load_object(model.pkl).predict(scaled_df)
         │
         ▼
 Predicted Math Score (e.g. 73.4)
         │
         ▼
 render_template('home.html', results=73.4)
         │
         ▼
 <h2>THE prediction is 73.4</h2>  ← displayed to user
```

---

## Results


##### Results available in Repo -> Results.pdf

---

```


*Built with [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/), [CatBoost](https://catboost.ai/), [Flask](https://flask.palletsprojects.com/), [Docker](https://www.docker.com/), AWS & Azure.*

*Built with 💖 by BikashBIOS*
