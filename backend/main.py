import numpy as np
from fastapi import FastAPI
from fastapi import BackgroundTasks
from urllib.parse import urlparse
import pandas as pd

import mlflow
from mlflow.tracking import MlflowClient
from ml.train import trainer
from ml.data import read_data, preprocessing, data_split, config
from ml.utils import scorer
from backend.models import TrainApiData, PredictApiData


cols = config().columns

mlflow.set_tracking_uri("sqlite:///db/backend.db")
app = FastAPI()
mlflowclient = MlflowClient(
    mlflow.get_tracking_uri(), mlflow.get_registry_uri())


def train_model_task(model_name: str, hyperparams: dict):

    mlflow.set_experiment("HR-Employee-Attrition")
    with mlflow.start_run() as run:
        

        # Prepare for training
        print("Loading data...")
        data = read_data("data/HR-Employee-Attrition.csv")
        data, encoder = preprocessing(data)
        x_train, x_test, y_train, y_test, train_cols, target = data_split(data)


        # Train
        print("Training model")
        params = {'bootstrap': True,
                 'ccp_alpha': 0.0,
                 'class_weight': hyperparams["class_weight"],
                 'criterion': hyperparams['criterion'],
                 'max_depth': None,
                 'max_features': hyperparams["max_features"],
                 'max_leaf_nodes': None,
                 'max_samples': None,
                 'min_impurity_decrease': 0.0,
                 'min_samples_leaf': 10,
                 'min_samples_split': 5,
                 'min_weight_fraction_leaf': 0.0,
                 'n_estimators': hyperparams['n_estimators'],
                 'n_jobs': -1,
                 'oob_score': False,
                 'random_state': 2021,
                 'verbose': 0,
                 'warm_start': False}
        
        model, preds, feature_df = trainer(x_train, y_train, x_test, y_test, train_cols, params)

        # Log hyperparameters
        mlflow.log_params(model.get_params())

        f1_pred, acc_pred, rec_pred, prec_pred = scorer(y_test, preds,is_return=True)        
        metrics = {'F1': f1_pred, 'Accuracy': acc_pred, 'Recall': rec_pred, 'Precision': prec_pred}
        
        print("Logging results")
        # Log in mlflow
        for metric in metrics.keys():
            mlflow.log_metric(metric, metrics[metric])


        # Register model
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        print(f"{tracking_url_type_store = }")
        print(model_name)

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                model, "RandomForest", registered_model_name=model_name, 
                conda_env=mlflow.sklearn.get_default_conda_env())
        else:
            mlflow.sklearn.log_model(
                model, "RandomForest-FS", registered_model_name=model_name)

        # Transition to production. We search for the last model with the name and we stage it to production
        mv = mlflowclient.search_model_versions(
            f"name='{model_name}'")[-1]  # Take last model version
        mlflowclient.transition_model_version_stage(
            name=mv.name, version=mv.version, stage="production")


@app.get("/")
async def main():
	return{"HealthCheck": "Still Alive!"}

@app.get("/uri")
async def read_root():
    return {"Tracking URI": mlflow.get_tracking_uri(),
            "Registry URI": mlflow.get_registry_uri()}


@app.get("/models")
async def get_models_api():
    """Gets a list with model names"""
    model_list = mlflowclient.list_registered_models()
    model_list = [model.name for model in model_list]
    return model_list


@app.post("/train")
async def train_api(data: TrainApiData, background_tasks: BackgroundTasks):
    """Creates a model based on hyperparameters and trains it."""
    hyperparams = data.hyperparams
    print(hyperparams)
    model_name = data.model_name
    print(model_name)

    background_tasks.add_task(
        train_model_task, model_name, hyperparams)

    return {"result": "Training task started"}


@app.post("/predict")
async def predict_api(data: PredictApiData):
    print(data)
    model_name = data.model_name
    df = pd.DataFrame([data.data], columns = cols)
    model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/Production")
    pred = np.round(model.predict_proba(df)[:, 1], 6)
    print(pred)
    return {"result": pred[0]}



@app.post("/importance")
async def importance_api(data: PredictApiData):
    model_name = data.model_name
    model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/Production")
    feature_df = pd.DataFrame(list(zip(cols, model.feature_importances_)), columns=["Feature", "Importance"])
    feature_df = feature_df.sort_values(by="Importance", ascending=False)
    
    return feature_df.to_dict()



