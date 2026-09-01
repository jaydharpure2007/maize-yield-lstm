import os
import time

import pandas as pd
import numpy as np
import torch
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

from src.data_loader import (
    get_excel_files,
    load_train_test
)

from src.preprocessing import (
    preprocess_data,
    remove_invalid_sequences
)

from src.feature_selection import (
    generate_uncorrelated_combinations,
    select_maximum_feature_combinations
)

from src.models import LSTMModel

from src.training import (
    prepare_dataloader,
    train_model
)

from src.optimization import (
    optimize_lstm
)

from src.evaluation import (
    evaluate_model
)

from src.utils import (
    set_random_seed,
    create_directory,
    get_best_trial
)

from config.settings import (
    FEATURE_GROUPS,
    FEATURE_GROUP_NAMES,
    TARGET_COLUMN,
    SAMPLE_ID_COLUMN,
    GROWTH_STAGE_COLUMN,
    CORRELATION_THRESHOLD,
    N_CV_SPLITS,
    CV_EPOCHS,
    N_OPTUNA_TRIALS,
    FINAL_MAX_EPOCHS,
    MODEL_DIRECTORY,
    RESULTS_DIRECTORY
)


def build_feature_list(
    selected_indices
):

    features = []

    for index in selected_indices:

        features.extend(
            FEATURE_GROUPS[index]
        )

    return features


def run_feature_selection_cv(
    train,
    factors,
    device,
    random_state
):

    print("\nFeature selection:")

    combinations_df = (
        generate_uncorrelated_combinations(
            train,
            factors,
            threshold=CORRELATION_THRESHOLD
        )
    )

    filtered_combinations, combination_names = (
        select_maximum_feature_combinations(
            combinations_df
        )
    )

    results = []

    kfold = KFold(
        n_splits=N_CV_SPLITS,
        shuffle=True,
        random_state=random_state
    )

    for group_name, selected_factors in (
        combination_names.items()
    ):

        print(
            f"  Evaluating {group_name}: "
            f"{len(selected_factors)} features"
        )

        data_model = train[
            [
                GROWTH_STAGE_COLUMN,
                SAMPLE_ID_COLUMN
            ]
            + selected_factors
            + [TARGET_COLUMN]
        ].copy()

        X, y, _ = preprocess_data(
            data_model
        )

        X, y, lengths = remove_invalid_sequences(
            X,
            y
        )

        scaler_y = MinMaxScaler()

        y_scaled = scaler_y.fit_transform(
            y
        )

        cv_rmse = []
        cv_r2 = []
        cv_mae = []

        for train_idx, val_idx in kfold.split(X):

            X_train = X[train_idx]
            X_val = X[val_idx]

            y_train = y_scaled[train_idx]
            y_val = y_scaled[val_idx]

            train_loader, _ = prepare_dataloader(
                X_train,
                y_train,
                batch_size=16
            )

            val_loader, _ = prepare_dataloader(
                X_val,
                y_val,
                batch_size=16
            )

            model = LSTMModel(
                input_size=X.shape[2],
                hidden_size=128,
                num_layers=1,
                dropout_rate=0.2
            ).to(device)

            criterion = torch.nn.MSELoss()

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=0.001
            )

            model, _ = train_model(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                max_epochs=CV_EPOCHS,
                patience=10
            )

            rmse, r2, mae, _, _ = evaluate_model(
                model=model,
                X=X_val,
                y=y_val,
                lengths=lengths[val_idx],
                scaler_y=scaler_y,
                device=device
            )

            cv_rmse.append(rmse)
            cv_r2.append(r2)
            cv_mae.append(mae)

        results.append(
            [
                group_name,
                selected_factors,
                np.mean(cv_r2),
                np.mean(cv_rmse),
                np.mean(cv_mae)
            ]
        )

    results_df = pd.DataFrame(
        results,
        columns=[
            "Factor Group",
            "Factors",
            "CV R2",
            "CV RMSE",
            "CV MAE"
        ]
    )

    results_df = results_df.sort_values(
        by=[
            "CV R2",
            "CV RMSE",
            "CV MAE"
        ],
        ascending=[
            False,
            True,
            True
        ]
    )

    best_row = results_df.iloc[0]

    best_factors = combination_names[
        best_row["Factor Group"]
    ]

    return (
        results_df,
        combination_names,
        best_factors
    )


def save_results(
    output_file,
    best_hyperparameters,
    trial_results,
    performance,
    feature_results,
    combinations_df,
    best_factors,
    train_results,
    test_results
):

    best_variables = pd.DataFrame(
        {
            "Best Variables": best_factors
        }
    )

    with pd.ExcelWriter(
        output_file
    ) as writer:

        best_hyperparameters.to_excel(
            writer,
            sheet_name="Best Hyperparameters",
            index=False
        )

        trial_results.to_excel(
            writer,
            sheet_name="Trial results",
            index=False
        )

        performance.to_excel(
            writer,
            sheet_name="Performance Metrics",
            index=False
        )

        feature_results.to_excel(
            writer,
            sheet_name="Factors result",
            index=False
        )

        combinations_df.to_excel(
            writer,
            sheet_name="Combinations",
            index=False
        )

        best_variables.to_excel(
            writer,
            sheet_name="Best Variables",
            index=False
        )

        train_results.to_excel(
            writer,
            sheet_name="Train results",
            index=False
        )

        test_results.to_excel(
            writer,
            sheet_name="Test results",
            index=False
        )


def run_experiment(
    working_directory,
    random_state,
    device
):

    set_random_seed(
        random_state
    )

    create_directory(
        MODEL_DIRECTORY
    )

    create_directory(
        RESULTS_DIRECTORY
    )

    excel_files = get_excel_files(
        working_directory
    )

    if not excel_files:

        raise FileNotFoundError(
            "No Excel files were found."
        )

    for file_path in excel_files:

        file_name = file_path.stem

        print(
            f"\n{'=' * 70}"
        )

        print(
            f"Processing: {file_name}"
        )

        print(
            f"{'=' * 70}"
        )

        train, test = load_train_test(
            file_path
        )

        for r in range(1, 5):

            from itertools import combinations

            for selected_indices in combinations(
                range(4),
                r
            ):

                selected_names = [
                    FEATURE_GROUP_NAMES[i]
                    for i in selected_indices
                ]

                folder_name = "_".join(
                    selected_names
                )

                output_dir = (
                    RESULTS_DIRECTORY
                    / folder_name
                )

                create_directory(
                    output_dir
                )

                combined_features = build_feature_list(
                    selected_indices
                )

                print(
                    f"\nCombination: "
                    f"{folder_name}"
                )

                print(
                    f"Total features: "
                    f"{len(combined_features)}"
                )

                factor_start = time.time()

                (
                    factors_results_df,
                    combination_names,
                    best_factors
                ) = run_feature_selection_cv(
                    train=train,
                    factors=combined_features,
                    device=device,
                    random_state=random_state
                )

                factor_time = (
                    time.time()
                    - factor_start
                )

                print(
                    "\nBest Factor Group:"
                )

                print(
                    factors_results_df.iloc[0]
                )

                # --------------------------------------------------
                # Prepare TRAIN data
                # --------------------------------------------------

                train_data = train[
                    [
                        GROWTH_STAGE_COLUMN,
                        SAMPLE_ID_COLUMN
                    ]
                    + best_factors
                    + [TARGET_COLUMN]
                ].copy()

                X_train, y_train, _ = (
                    preprocess_data(
                        train_data
                    )
                )

                X_train, y_train, train_lengths = (
                    remove_invalid_sequences(
                        X_train,
                        y_train
                    )
                )

                scaler_y_train = MinMaxScaler()

                y_train_scaled = (
                    scaler_y_train.fit_transform(
                        y_train
                    )
                )

                X_train = X_train.astype(
                    np.float32
                )

                # --------------------------------------------------
                # Prepare TEST data
                # --------------------------------------------------

                test_data = test[
                    [
                        GROWTH_STAGE_COLUMN,
                        SAMPLE_ID_COLUMN
                    ]
                    + best_factors
                    + [TARGET_COLUMN]
                ].copy()

                X_test, y_test, _ = (
                    preprocess_data(
                        test_data
                    )
                )

                X_test, y_test, test_lengths = (
                    remove_invalid_sequences(
                        X_test,
                        y_test
                    )
                )

                scaler_y_test = MinMaxScaler()

                y_test_scaled = (
                    scaler_y_test.fit_transform(
                        y_test
                    )
                )

                X_test = X_test.astype(
                    np.float32
                )

                # --------------------------------------------------
                # OPTUNA
                # --------------------------------------------------

                model_start = time.time()

                study = optimize_lstm(
                    X_train=X_train,
                    y_train=y_train_scaled,
                    scaler_y=scaler_y_train,
                    device=device,
                    n_trials=N_OPTUNA_TRIALS,
                    random_state=random_state
                )

                best_trial = get_best_trial(
                    study
                )

                best_params = best_trial.params

                print(
                    "\nBest Optuna parameters:"
                )

                print(
                    best_params
                )

                # --------------------------------------------------
                # Final model
                # --------------------------------------------------

                final_model = LSTMModel(
                    input_size=X_train.shape[2],
                    hidden_size=int(
                        best_params["hidden_size"]
                    ),
                    num_layers=int(
                        best_params["num_layers"]
                    ),
                    dropout_rate=float(
                        best_params["dropout_rate"]
                    )
                ).to(device)

                batch_size = int(
                    best_params["batch_size"]
                )

                learning_rate = float(
                    best_params["lr"]
                )

                patience = int(
                    best_params["patience"]
                )

                train_loader, _ = prepare_dataloader(
                    X_train,
                    y_train_scaled,
                    batch_size
                )

                test_loader, _ = prepare_dataloader(
                    X_test,
                    y_test_scaled,
                    batch_size
                )

                criterion = torch.nn.MSELoss()

                optimizer = torch.optim.Adam(
                    final_model.parameters(),
                    lr=learning_rate
                )

                final_model, final_epochs = train_model(
                    model=final_model,
                    criterion=criterion,
                    optimizer=optimizer,
                    train_loader=train_loader,
                    val_loader=test_loader,
                    device=device,
                    max_epochs=FINAL_MAX_EPOCHS,
                    patience=patience
                )

                # --------------------------------------------------
                # Evaluation
                # --------------------------------------------------

                (
                    train_rmse,
                    train_r2,
                    train_mae,
                    y_train_obs,
                    y_train_pred
                ) = evaluate_model(
                    final_model,
                    X_train,
                    y_train_scaled,
                    train_lengths,
                    scaler_y_train,
                    device
                )

                (
                    test_rmse,
                    test_r2,
                    test_mae,
                    y_test_obs,
                    y_test_pred
                ) = evaluate_model(
                    final_model,
                    X_test,
                    y_test_scaled,
                    test_lengths,
                    scaler_y_test,
                    device
                )

                print(
                    f"\nTraining R²: {train_r2:.4f}"
                )

                print(
                    f"Testing R²: {test_r2:.4f}"
                )

                # --------------------------------------------------
                # Save model
                # --------------------------------------------------

                model_file = (
                    output_dir
                    / f"{file_name}_final_lstm_model.pt"
                )

                torch.save(
                    {
                        "model_state_dict":
                            final_model.state_dict(),

                        "input_size":
                            X_train.shape[2],

                        "hidden_size":
                            int(
                                best_params[
                                    "hidden_size"
                                ]
                            ),

                        "num_layers":
                            int(
                                best_params[
                                    "num_layers"
                                ]
                            ),

                        "dropout_rate":
                            float(
                                best_params[
                                    "dropout_rate"
                                ]
                            ),

                        "features":
                            best_factors
                    },
                    model_file
                )

                # --------------------------------------------------
                # Results
                # --------------------------------------------------

                performance_df = pd.DataFrame(
                    [
                        {
                            "Dataset": "Training",
                            "RMSE": train_rmse,
                            "MAE": train_mae,
                            "R2": train_r2
                        },
                        {
                            "Dataset": "Testing",
                            "RMSE": test_rmse,
                            "MAE": test_mae,
                            "R2": test_r2
                        }
                    ]
                )

                trial_results = []

                for trial in study.trials:

                    trial_results.append(
                        {
                            "Trial": trial.number,
                            "RMSE": trial.value,
                            "R2":
                                trial.user_attrs.get(
                                    "R2"
                                ),
                            "Best_Epoch":
                                trial.user_attrs.get(
                                    "max_best_epoch"
                                ),
                            **trial.params
                        }
                    )

                trials_df = pd.DataFrame(
                    trial_results
                )

                trials_df = trials_df.sort_values(
                    by="R2",
                    ascending=False
                )

                best_hyperparameters = (
                    pd.DataFrame(
                        [best_params]
                    )
                )

                train_results = pd.DataFrame(
                    {
                        "Observed":
                            y_train_obs,

                        "Predicted":
                            y_train_pred
                    }
                )

                test_results = pd.DataFrame(
                    {
                        "Observed":
                            y_test_obs,

                        "Predicted":
                            y_test_pred
                    }
                )

                output_file = (
                    output_dir
                    / f"{file_name}_LSTM_CV_Results_PyTorch.xlsx"
                )

                save_results(
                    output_file=output_file,
                    best_hyperparameters=
                        best_hyperparameters,
                    trial_results=trials_df,
                    performance=performance_df,
                    feature_results=
                        factors_results_df,
                    combinations_df=
                        pd.DataFrame(
                            combination_names
                        ),
                    best_factors=best_factors,
                    train_results=train_results,
                    test_results=test_results
                )

                processing_time = (
                    time.time()
                    - model_start
                )

                print(
                    f"\nFactor selection time: "
                    f"{factor_time:.2f} seconds"
                )

                print(
                    f"Model processing time: "
                    f"{processing_time:.2f} seconds"
                )

                print(
                    f"Saved: {output_file}"
                )