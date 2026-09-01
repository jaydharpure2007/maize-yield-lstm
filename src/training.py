import copy

import numpy as np
import torch

import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader
)


def prepare_dataloader(
    X,
    y,
    batch_size
):

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

    valid = lengths > 0

    X = X[valid]
    y = y[valid]
    lengths = lengths[valid]

    X_tensor = torch.from_numpy(
        X
    ).float()

    y_tensor = torch.from_numpy(
        y
    ).float()

    lengths_tensor = torch.from_numpy(
        lengths
    ).long()

    dataset = TensorDataset(
        X_tensor,
        lengths_tensor,
        y_tensor
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return loader, lengths


def train_model(
    model,
    criterion,
    optimizer,
    train_loader,
    val_loader,
    device,
    max_epochs=1000,
    patience=10
):

    best_loss = float("inf")

    best_state = None

    best_epoch = 0

    counter = 0

    for epoch in range(max_epochs):

        # --------------------------------------------------
        # Training
        # --------------------------------------------------

        model.train()

        for xb, lengths, yb in train_loader:

            xb = xb.to(device)
            lengths = lengths.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            predictions = model(
                xb,
                lengths
            )

            loss = criterion(
                predictions,
                yb
            )

            loss.backward()

            optimizer.step()

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        model.eval()

        validation_losses = []

        with torch.no_grad():

            for xb, lengths, yb in val_loader:

                xb = xb.to(device)
                lengths = lengths.to(device)
                yb = yb.to(device)

                predictions = model(
                    xb,
                    lengths
                )

                loss = criterion(
                    predictions,
                    yb
                )

                validation_losses.append(
                    loss.item()
                )

        validation_loss = np.mean(
            validation_losses
        )

        # --------------------------------------------------
        # Early stopping
        # --------------------------------------------------

        if validation_loss < best_loss:

            best_loss = validation_loss

            best_state = copy.deepcopy(
                model.state_dict()
            )

            best_epoch = epoch + 1

            counter = 0

        else:

            counter += 1

        if counter >= patience:

            break

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    return model, best_epoch