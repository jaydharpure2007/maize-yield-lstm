from config.settings import (
    WORKING_DIRECTORY,
    RANDOM_STATE,
    DEVICE
)

from src.experiment import run_experiment


def main():
    print("=" * 70)
    print("MAIZE YIELD PREDICTION USING LSTM")
    print("=" * 70)

    print(f"Working directory: {WORKING_DIRECTORY}")
    print(f"Device: {DEVICE}")

    run_experiment(
        working_directory=WORKING_DIRECTORY,
        random_state=RANDOM_STATE,
        device=DEVICE
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()