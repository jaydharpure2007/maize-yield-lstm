import numpy as np

from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler
)


def encode_growth_stage(data):

    data = data.copy()

    encoder = LabelEncoder()

    data["GS_Encoded"] = (
        encoder.fit_transform(data["GS1"]) + 1
    )

    return data, encoder


def scale_features(data):

    data = data.copy()

    feature_columns = data.columns[
        2:len(data.columns) - 2
    ]

    scaler = MinMaxScaler()

    data.loc[:, feature_columns] = (
        scaler.fit_transform(
            data[feature_columns]
        )
    )

    return data, scaler


def build_sequences(data):

    grouped = data.groupby(
        "Sample_ID",
        sort=False
    )

    sequences = []
    yields = []

    for _, group in grouped:

        sequence = group.iloc[:, 2:-1].values

        target = group["Yield"].iloc[-1]

        sequences.append(sequence)

        yields.append(target)

    if not sequences:
        raise ValueError(
            "No valid sequences were created."
        )

    max_length = max(
        len(sequence)
        for sequence in sequences
    )

    X = np.array(
        [
            np.pad(
                sequence,
                (
                    (0, max_length - len(sequence)),
                    (0, 0)
                ),
                mode="constant"
            )
            for sequence in sequences
        ],
        dtype=np.float32
    )

    y = np.asarray(
        yields,
        dtype=np.float32
    ).reshape(-1, 1)

    return X, y


def preprocess_data(data_model):

    data_model = data_model.copy()

    data_model, _ = encode_growth_stage(
        data_model
    )

    data_model, feature_scaler = scale_features(
        data_model
    )

    ordered_columns = (
        list(data_model.columns[:2])
        + ["GS_Encoded"]
        + list(data_model.columns[2:-1])
    )

    data_model = data_model[
        ordered_columns
    ]

    X, y = build_sequences(
        data_model
    )

    return X, y, feature_scaler


def calculate_sequence_lengths(X):

    lengths = np.array(
        [
            np.sum(
                ~np.all(
                    sequence == 0,
                    axis=-1
                )
            )
            for sequence in X
        ],
        dtype=np.int64
    )

    return lengths


def remove_invalid_sequences(X, y):

    lengths = calculate_sequence_lengths(X)

    valid = lengths > 0

    X = X[valid]
    y = y[valid]
    lengths = lengths[valid]

    return X, y, lengths