from pathlib import Path
import torch

# PROJECT_ROOT = Path(__file__).resolve().parent.parent

# WORKING_DIRECTORY = PROJECT_ROOT / "data"


# ============================================================
# PROJECT PATHS
# ============================================================

WORKING_DIRECTORY = Path(
    r"C:\Users\jaydeo.dharpure\OneDrive - Texas A&M AgriLife\Python code\Bala work\LSTM code"
)

DATA_DIRECTORY = WORKING_DIRECTORY

OUTPUT_DIRECTORY = WORKING_DIRECTORY / "outputs"

RESULTS_DIRECTORY = OUTPUT_DIRECTORY / "results"

MODEL_DIRECTORY = OUTPUT_DIRECTORY / "models"

LOG_DIRECTORY = OUTPUT_DIRECTORY / "logs"


# ============================================================
# RANDOM SEED
# ============================================================

RANDOM_STATE = 42


# ============================================================
# COMPUTING DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CROSS-VALIDATION
# ============================================================

N_CV_SPLITS = 5


# ============================================================
# INITIAL CV LSTM
# ============================================================

CV_EPOCHS = 100


# ============================================================
# OPTUNA
# ============================================================

N_OPTUNA_TRIALS = 1

OPTUNA_DIRECTION = "minimize"


# ============================================================
# FINAL MODEL
# ============================================================

FINAL_MAX_EPOCHS = 1000


# ============================================================
# FEATURE CORRELATION
# ============================================================

CORRELATION_THRESHOLD = 0.75


# ============================================================
# FEATURE GROUPS
# ============================================================

WD = [
    "CPPT_cm",
    "CR_MJ/m2",
    "C_GDD"
]

SD = [
    "EC_30",
    "EC_60",
    "EC_90"
]

SF = [
    "CC",
    "CH"
]

RS = [
    "blue475",
    "green560",
    "red668",
    "rededge717",
    "nir842",
    "NDVI",
    "NGRDI",
    "GNDVI",
    "NDRE",
    "EVI",
    "SAVI",
    "MSAVI",
    "TVI",
    "RTVIcore",
    "VARI",
    "PSRI"
]

FEATURE_GROUPS = [
    WD,
    SD,
    SF,
    RS
]

FEATURE_GROUP_NAMES = [
    "WD",
    "SD",
    "SF",
    "RS"
]


# ============================================================
# TARGET
# ============================================================

TARGET_COLUMN = "Yield"

SAMPLE_ID_COLUMN = "Sample_ID"

GROWTH_STAGE_COLUMN = "GS1"


# ============================================================
# OPTUNA SEARCH SPACE
# ============================================================

OPTUNA_HIDDEN_SIZE = (50, 300)

OPTUNA_NUM_LAYERS = (1, 3)

OPTUNA_DROPOUT = (0.1, 0.4)

OPTUNA_BATCH_SIZE = (8, 64)

OPTUNA_LR = (1e-4, 1e-2)

OPTUNA_PATIENCE = (10, 20)