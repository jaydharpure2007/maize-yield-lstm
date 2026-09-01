import torch
import torch.nn as nn

from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence
)


class LSTMModel(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        dropout_rate
    ):

        super().__init__()

        # PyTorch ignores dropout for a single LSTM layer.
        lstm_dropout = (
            dropout_rate
            if num_layers > 1
            else 0.0
        )

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout
        )

        self.dropout = nn.Dropout(
            dropout_rate
        )

        self.fc1 = nn.Linear(
            hidden_size,
            15
        )

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(
            15,
            1
        )

    def forward(
        self,
        x,
        lengths
    ):

        packed_input = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_output, _ = self.lstm(
            packed_input
        )

        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True
        )

        batch_indices = torch.arange(
            x.size(0),
            device=x.device
        )

        last_outputs = output[
            batch_indices,
            lengths - 1
        ]

        x = self.dropout(
            last_outputs
        )

        x = self.relu(
            self.fc1(x)
        )

        return self.fc2(x)

    def predict(
        self,
        X,
        lengths
    ):

        self.eval()

        with torch.no_grad():

            return self.forward(
                X,
                lengths
            )