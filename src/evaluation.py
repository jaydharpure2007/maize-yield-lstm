import numpy as np

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

import torch


def evaluate_model(
    model,
    X,
    y,
    lengths,
    scaler_y,
    device
):

    model.eval()

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=device
    )

    lengths_tensor = torch.tensor(
        lengths,
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():

        predictions = model(
            X_tensor,
            lengths_tensor
        ).cpu().numpy()

    predictions = scaler_y.inverse_transform(
        predictions
    )

    observations = scaler_y.inverse_transform(
        y
    )

    rmse = np.sqrt(
        mean_squared_error(
            observations,
            predictions
        )
    )

    mae = mean_absolute_error(
        observations,
        predictions
    )

    r2 = r2_score(
        observations,
        predictions
    )

    return (
        rmse,
        r2,
        mae,
        observations.flatten(),
        predictions.flatten()
    )