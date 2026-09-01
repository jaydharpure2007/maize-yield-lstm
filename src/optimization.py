import numpy as np
import optuna
import torch

from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_squared_error,
    r2_score
)

from sklearn.preprocessing import MinMaxScaler

from src.models import LSTMModel
from src.training import (
    prepare_dataloader,
    train_model
)


def create_objective(
    X_train,
    y_train,
    scaler_y,
    device,
    random_state=42
):

    def objective(trial):

        hidden_size = trial.suggest_int(
            "hidden_size",
            50,
            300
        )

        num_layers = trial.suggest_int(
            "num_layers",
            1,
            3
        )

        dropout_rate = trial.suggest_float(
            "dropout_rate",
            0.1,
            0.4
        )

        batch_size = trial.suggest_int(
            "batch_size",
            8,
            64
        )

        learning_rate = trial.suggest_float(
            "lr",
            1e-4,
            1e-2,
            log=True
        )

        patience = trial.suggest_int(
            "patience",
            10,
            20
        )

        kfold = KFold(
            n_splits=5,
            shuffle=True,
            random_state=random_state
        )

        rmses = []

        r2_scores = []

        best_epochs = []

        for train_idx, val_idx in kfold.split(
            X_train
        ):

            X_tr = X_train[train_idx]
            X_val = X_train[val_idx]

            y_tr = y_train[train_idx]
            y_val = y_train[val_idx]

            train_loader, _ = prepare_dataloader(
                X_tr,
                y_tr,
                batch_size
            )

            val_loader, _ = prepare_dataloader(
                X_val,
                y_val,
                batch_size
            )

            model = LSTMModel(
                input_size=X_train.shape[2],
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout_rate=dropout_rate
            ).to(device)

            criterion = torch.nn.MSELoss()

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=learning_rate
            )

            model, best_epoch = train_model(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                max_epochs=1000,
                patience=patience
            )

            predictions = []
            targets = []

            model.eval()

            with torch.no_grad():

                for xb, lengths, yb in val_loader:

                    pred = model(
                        xb.to(device),
                        lengths.to(device)
                    )

                    predictions.append(
                        pred.cpu().numpy()
                    )

                    targets.append(
                        yb.numpy()
                    )

            predictions = np.concatenate(
                predictions
            )

            targets = np.concatenate(
                targets
            )

            predictions_inv = (
                scaler_y.inverse_transform(
                    predictions
                )
            )

            targets_inv = (
                scaler_y.inverse_transform(
                    targets
                )
            )

            rmse = np.sqrt(
                mean_squared_error(
                    targets_inv,
                    predictions_inv
                )
            )

            r2 = r2_score(
                targets_inv,
                predictions_inv
            )

            rmses.append(rmse)

            r2_scores.append(r2)

            best_epochs.append(
                best_epoch
            )

        mean_rmse = np.mean(rmses)

        mean_r2 = np.mean(r2_scores)

        max_best_epoch = np.max(
            best_epochs
        )

        trial.set_user_attr(
            "R2",
            mean_r2
        )

        trial.set_user_attr(
            "max_best_epoch",
            int(max_best_epoch)
        )

        return mean_rmse

    return objective


def optimize_lstm(
    X_train,
    y_train,
    scaler_y,
    device,
    n_trials=30,
    random_state=42
):

    objective = create_objective(
        X_train=X_train,
        y_train=y_train,
        scaler_y=scaler_y,
        device=device,
        random_state=random_state
    )

    sampler = optuna.samplers.TPESampler(
        seed=random_state
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler
    )

    study.optimize(
        objective,
        n_trials=n_trials
    )

    return study