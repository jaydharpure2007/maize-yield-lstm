from pathlib import Path
import pandas as pd


def get_excel_files(data_directory):
    """
    Return all Excel files in the data directory.
    """
    DATA_DIR = data_directory / "data"
    data_directory = Path(DATA_DIR)

    return sorted(
        data_directory.glob("*.xlsx")
    )


def load_train_test(file_path):
    """
    Load train and test sheets from an Excel workbook.
    """

    train = pd.read_excel(
        file_path,
        sheet_name="train"
    )

    test = pd.read_excel(
        file_path,
        sheet_name="test"
    )

    return train, test