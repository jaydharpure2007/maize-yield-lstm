import torch

from torch.utils.data import Dataset


class SequenceDataset(Dataset):

    def __init__(
        self,
        sequences,
        targets,
        lengths
    ):

        self.sequences = torch.tensor(
            sequences,
            dtype=torch.float32
        )

        self.targets = torch.tensor(
            targets,
            dtype=torch.float32
        )

        self.lengths = torch.tensor(
            lengths,
            dtype=torch.int64
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):

        return (
            self.sequences[idx],
            self.lengths[idx],
            self.targets[idx]
        )