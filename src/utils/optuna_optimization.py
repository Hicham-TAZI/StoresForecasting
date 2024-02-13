# File: ForecastProject/src/utils/optuna_optimization.py

import optuna
from darts.metrics import rmse
from src.models.tcn_model import build_fit_tcn_model

def objective(trial, series, out_len=100):
    """
    Objective function for Optuna optimization, tailored for TCN model.
    """
    # Hyperparameters to be tuned
    kernel_size = trial.suggest_int("kernel_size", 2, 25)
    num_filters = trial.suggest_int("num_filters", 5, 50)
    dilation_base = trial.suggest_int("dilation_base", 2, 4)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    include_dayofweek = trial.suggest_categorical("include_dayofweek", [True, False])
    in_len = trial.suggest_int("in_len", 101,200) 
    out_len = trial.suggest_int("out_len", 100,100) 


    # Build and fit the TCN model
    model, train_scaled, val_scaled, _ = build_fit_tcn_model(
        series=series,
        in_len=in_len,
        out_len=out_len,
        kernel_size=kernel_size,
        num_filters=num_filters,
        dilation_base=dilation_base,
        dropout=dropout,
        lr=lr,
        include_dayofweek=include_dayofweek,
        callbacks=None  # Assuming callbacks are managed within the model function
    )

    # Predict on the validation set and compute RMSE
    val_pred = model.predict(series=train_scaled, n=out_len)
    val_rmse = rmse(val_scaled, val_pred)

    return val_rmse

def optimize_model(series, n_trials=100):
    """
    Runs the Optuna optimization.
    """
    # Define the range for input_chunk_length as a fixed parameter for this optimization
    
    def objective_wrapper(trial):
        # Extract in_len within the wrapper to use it as a dynamic parameter
        # set out_len, between 1 and 13 days (it has to be strictly shorter than in_len).
        days_out = 100
        out_len = days_out 
        return objective(trial, series, out_len)
        
    study = optuna.create_study(direction="minimize")
    study.optimize(objective_wrapper, n_trials=n_trials)

    print(f"Best RMSE: {study.best_value}")
    print(f"Best params: {study.best_params}")

    return study.best_params
