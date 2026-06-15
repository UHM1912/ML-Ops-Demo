# MLOps Demo Presenter Notes

## 1. What this project is

This repo is an end-to-end MLOps demo for an insurance purchase prediction use case.

It shows:

- code versioning with Git
- data versioning with DVC
- model training and experiment tracking with MLflow
- model serving through FastAPI
- drift reporting with Evidently
- CI validation with GitHub Actions

The frontend exists only as a thin UI layer. The real MLOps logic is in `backend/`.

## 2. Business problem

The model predicts whether a customer is likely to purchase insurance.

Target column:

- `Result`

Input features used after cleaning:

- `Gender`
- `Age`
- `HasDrivingLicense`
- `RegionID`
- `Switch`
- `PastAccident`
- `AnnualPremium`

## 3. Backend flow in simple words

### Training flow

1. Load train and test CSV files from `backend/data/`
2. Clean and standardize the data
3. Split features and target
4. Apply preprocessing
5. Handle class imbalance using SMOTE
6. Train the configured model
7. Save the trained pipeline as `backend/models/model.pkl`
8. Evaluate on test data
9. Log params, metrics, and model artifact to MLflow
10. Attempt to register the model as `insurance_model`

### Serving flow

The FastAPI app loads the latest saved model on startup and exposes endpoints for:

- health check
- prediction
- retraining
- experiment history
- drift reporting
- chat-based routing

## 4. Important files to explain

- `backend/main.py`: training pipeline orchestration
- `backend/app.py`: API layer and operational endpoints
- `backend/steps/ingest.py`: load train/test data from config
- `backend/steps/clean.py`: cleaning and feature preparation
- `backend/steps/train.py`: preprocessing, SMOTE, model training, save model
- `backend/steps/predict.py`: model loading and evaluation
- `backend/config.yml`: model choice, hyperparameters, data paths
- `data.dvc`: DVC pointer for versioned data folder
- `.github/workflows/mlops-ci-cd.yml`: CI pipeline

## 5. How each MLOps fundamental appears in this repo

### Version control

Git is used for:

- code history
- config history
- pipeline evolution
- CI workflow tracking

What to say:

"Git versions the code and configuration, so we can always trace when a logic change, bug fix, or pipeline update was introduced."

### Data versioning

DVC is used at repo root.

Evidence:

- `.dvc/`
- `data.dvc`
- `backend/data/` tracked as a DVC output

What to say:

"Git is not ideal for large datasets, so DVC keeps lightweight metadata in Git and stores actual data in remote storage. That gives us dataset reproducibility without bloating the repo."

### Experiment tracking

MLflow is used in `backend/main.py`.

It logs:

- model parameters
- accuracy
- ROC AUC
- precision
- recall
- tags such as developer and preprocessing details

What to say:

"Instead of manually remembering which model performed best, MLflow stores each run with its parameters and metrics, so experiments become comparable and reproducible."

### Model registry

The code attempts to register the trained model under:

- `insurance_model`

What to say:

"The registry is the controlled catalog of trained model versions. It is where we move from just training models to managing model lifecycle."


## 6. What exactly happens in preprocessing

Cleaning logic in `backend/steps/clean.py`:

- drops rows with missing target `Result`
- drops columns: `id`, `SalesChannelID`, `VehicleAge`, `DaysSinceCreated`
- removes currency symbols from `AnnualPremium` and converts to float
- imputes `Gender` and `RegionID` using most frequent value
- fills missing `Age` with median
- fills missing `HasDrivingLicense` with `1`
- fills missing `Switch` with `-1`
- fills missing `PastAccident` with `Unknown`
- removes outliers from `AnnualPremium` using IQR upper bound

Preprocessing in `backend/steps/train.py`:

- `MinMaxScaler` on `AnnualPremium`
- `StandardScaler` on `Age` and `RegionID`
- `OneHotEncoder` on `Gender` and `PastAccident`
- `SMOTE` for class balancing
- classifier chosen from config

Current configured model:

- `DecisionTreeClassifier`
- criterion: `entropy`

## 7. API endpoints you can demo

- `GET /health`
- `POST /predict`
- `POST /train`
- `GET /experiments`
- `POST /drift`
- `POST /chat`

## 8. Best demo sequence

1. Start with the architecture story
2. Show Git and DVC files at repo root
3. Open `backend/config.yml` and say model choice is config-driven
4. Open `backend/main.py` and explain the training flow
5. Open `backend/steps/clean.py` and `backend/steps/train.py`
6. Show `backend/app.py` and map endpoints to MLOps stages
7. Trigger training
8. Show MLflow experiment tracking
9. Show prediction endpoint
10. Show drift report generation
11. Close by tying everything back to lifecycle management

## 9. One-line explanation for each endpoint

- `/train`: retrains the full ML pipeline and logs the run to MLflow
- `/experiments`: fetches recent MLflow runs and their metrics
- `/predict`: serves inference from the latest saved model
- `/drift`: compares training data and production-like data using Evidently
- `/chat`: converts user intent into backend actions like training or prediction


