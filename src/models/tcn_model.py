import torch
from darts.dataprocessing.transformers import Scaler
from pytorch_lightning.callbacks import EarlyStopping
from darts.models import TCNModel

def build_fit_tcn_model(
    series,
    in_len,
    out_len,
    kernel_size,
    num_filters,
    dilation_base,
    dropout,
    lr,
    include_dayofweek,
    likelihood=None,
    callbacks=None,
):
    # Ensure reproducibility
    torch.manual_seed(42)

    val_len = out_len

    # Split series into train, validation, and scale them
    train = series[: -(2 * val_len)]
    val = series[-(2 * val_len) : -val_len]
    test = series[-val_len:]

    # Scale the series
    scaler = Scaler()
    train_scaled = scaler.fit_transform(train)
    val_scaled = scaler.transform(val)
    test_scaled = scaler.transform(test)

    model_val_set = scaler.transform(
        series[-((2 * val_len) + in_len) : -val_len]
    )
    
    # Define fixed parameters for all models
    BATCH_SIZE = 128
    MAX_N_EPOCHS = 100
    NR_EPOCHS_VAL_PERIOD = 1
    MAX_SAMPLES_PER_TS = 10

    # Setup early stopping
    early_stopper = EarlyStopping("val_loss", min_delta=0.001, patience=10, verbose=True)
    callbacks = [early_stopper] if callbacks is None else [early_stopper] + callbacks

    # GPU availability check
    pl_trainer_kwargs = {"accelerator": "gpu", "devices": 1, "callbacks": callbacks} if torch.cuda.is_available() else {"callbacks": callbacks}

    # Optionally add day of the week as a past covariate
    encoders = {"cyclic": {"past": ["dayofweek"]}} if include_dayofweek else None
    
    # Build the TCN model
    model = TCNModel(
        input_chunk_length=in_len,
        output_chunk_length=out_len,
        kernel_size=kernel_size,
        num_filters=num_filters,
        dilation_base=dilation_base,
        dropout=dropout,
        optimizer_kwargs={"lr": lr},
        batch_size=BATCH_SIZE,
        n_epochs=MAX_N_EPOCHS,
        nr_epochs_val_period=NR_EPOCHS_VAL_PERIOD,
        add_encoders=encoders,
        likelihood=likelihood,
        model_name="TCN_optuna_model",
        pl_trainer_kwargs=pl_trainer_kwargs,
        force_reset=True,
        save_checkpoints=True,
    )

    # Train the model
    model.fit(series=train_scaled, val_series=model_val_set, verbose=True)

    return model, train_scaled, val_scaled, test_scaled
