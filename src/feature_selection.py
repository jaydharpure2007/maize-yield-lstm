from itertools import combinations

import pandas as pd


def calculate_pairwise_correlations(
    data,
    features
):

    correlations = {}

    for i, feature_1 in enumerate(features):

        for feature_2 in features[i + 1:]:

            correlations[
                (feature_1, feature_2)
            ] = data[feature_1].corr(
                data[feature_2]
            )

    return correlations


def is_correlated(
    combination,
    correlation_pairwise,
    threshold
):

    for feature_1, feature_2 in combinations(
        combination,
        2
    ):

        correlation = correlation_pairwise.get(
            (feature_1, feature_2)
        )

        if correlation is None:

            correlation = correlation_pairwise.get(
                (feature_2, feature_1),
                0
            )

        if abs(correlation) > threshold:

            return True

    return False


def generate_uncorrelated_combinations(
    data,
    features,
    threshold=0.75
):

    correlations = calculate_pairwise_correlations(
        data,
        features
    )

    combinations_list = []

    for size in range(
        1,
        len(features) + 1
    ):

        for combination in combinations(
            features,
            size
        ):

            if not is_correlated(
                combination,
                correlations,
                threshold
            ):

                combinations_list.append(
                    combination
                )

    if not combinations_list:

        raise ValueError(
            "No uncorrelated feature combinations found."
        )

    combinations_df = pd.DataFrame(
        {
            "Name": [
                f"C{i + 1}"
                for i in range(
                    len(combinations_list)
                )
            ],
            "Factors": [
                ", ".join(combo)
                for combo in combinations_list
            ]
        }
    )

    return combinations_df


def select_maximum_feature_combinations(
    combinations_df
):

    combinations_df = combinations_df.copy()

    combinations_df["n_features"] = (
        combinations_df["Factors"]
        .str.split(", ")
        .str.len()
    )

    max_features = (
        combinations_df["n_features"].max()
    )

    filtered = combinations_df[
        combinations_df["n_features"]
        == max_features
    ].copy()

    filtered = filtered.drop(
        columns=["n_features"]
    )

    combination_names = {
        f"C{i + 1}": row["Factors"].split(", ")
        for i, (_, row)
        in enumerate(filtered.iterrows())
    }

    return filtered, combination_names