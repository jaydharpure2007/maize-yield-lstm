import os
import random

import numpy as np
import torch


def set_random_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False


def create_directory(path):

    os.makedirs(
        path,
        exist_ok=True
    )


def get_best_trial(study):

    valid_trials = [
        trial
        for trial in study.trials
        if trial.user_attrs.get("R2") is not None
    ]

    if not valid_trials:

        raise ValueError(
            "No valid Optuna trials found."
        )

    best_trial = max(
        valid_trials,
        key=lambda trial:
            trial.user_attrs["R2"]
    )

    return best_trial