# Growth Stage-Specific Maize Yield Prediction Using LSTM

A PyTorch-based deep learning framework for **growth stage-specific maize yield prediction** using UAV multispectral observations and multi-source environmental and crop-related data.

The framework integrates weather, soil, structural, and remote sensing features into sequential observations and uses a Long Short-Term Memory (LSTM) neural network to model the temporal progression of crop growth.

> **Note:** This repository contains the modeling code and computational workflow. The experimental Excel datasets are not currently distributed because the associated research is under publication/review.

---

## 1. Overview

Accurate maize yield prediction requires information about crop development throughout the growing season. This project uses growth-stage observations to construct sequential inputs for an LSTM model.

The framework evaluates different combinations of feature groups, performs correlation-based feature selection, identifies suitable model configurations through cross-validation and Optuna optimization, and evaluates the final model using an independent test dataset.

### Main objectives

* Integrate UAV multispectral and multi-source crop/environmental data.
* Represent crop development as sequential growth-stage observations.
* Evaluate different combinations of feature groups.
* Reduce redundancy using correlation-based feature selection.
* Optimize LSTM hyperparameters using Optuna.
* Evaluate prediction performance using independent test data.
* Provide a modular and reproducible research codebase.

---

## 2. Modeling Workflow

The overall modeling workflow consists of the following steps:

```text
Input Excel Data
       │
       ▼
Data Loading
       │
       ▼
Data Preprocessing
       │
       ├── Growth-stage encoding
       ├── Feature scaling
       └── Sequence construction
       │
       ▼
Feature Group Combination
       │
       ▼
Correlation-Based Feature Selection
       │
       ▼
Cross-Validation
       │
       ▼
LSTM Model Training
       │
       ▼
Optuna Hyperparameter Optimization
       │
       ▼
Final Model Training
       │
       ▼
Independent Test Evaluation
       │
       ▼
Performance Results
```

### Workflow steps

1. Load training and testing data from Excel files.
2. Encode growth-stage information.
3. Scale model input features using Min-Max normalization.
4. Construct sequential observations for each sample.
5. Generate combinations of predefined feature groups.
6. Remove highly correlated variables.
7. Evaluate candidate feature sets using cross-validation.
8. Train LSTM models using the selected feature set.
9. Optimize LSTM hyperparameters using Optuna.
10. Train the final model using the selected configuration.
11. Evaluate the final model using an independent test dataset.
12. Save model performance, optimized parameters, and prediction results.

---

## 3. Feature Groups

Four major groups of predictor variables are considered.

| Group  | Description                                      | Examples                              |
| ------ | ------------------------------------------------ | ------------------------------------- |
| **WD** | Weather-derived variables                        | CPPT, CR, GDD                         |
| **SD** | Soil-derived variables                           | EC at different depths                |
| **SF** | Structural/crop variables                        | Canopy cover, canopy height           |
| **RS** | UAV multispectral and vegetation-index variables | Spectral bands and vegetation indices |

### Feature groups

#### WD — Weather-Derived Variables

```text
CPPT_cm
CR_MJ/m2
C_GDD
```

#### SD — Soil-Derived Variables

```text
EC_30
EC_60
EC_90
```

#### SF — Structural/Crop Variables

```text
CC
CH
```

#### RS — Remote Sensing Variables

```text
blue475
green560
red668
rededge717
nir842

NDVI
NGRDI
GNDVI
NDRE
EVI
SAVI
MSAVI
TVI
RTVIcore
VARI
PSRI
```

All possible non-empty combinations of the four feature groups are considered during feature-group evaluation.

---

## 4. LSTM Model

The prediction model is implemented using **PyTorch** and is designed to process sequential growth-stage observations.

The LSTM architecture consists of:

```text
Sequential Growth-Stage Features
              │
              ▼
          LSTM Layers
              │
              ▼
           Dropout
              │
              ▼
      Fully Connected Layer
              │
              ▼
             ReLU
              │
              ▼
      Fully Connected Layer
              │
              ▼
        Predicted Yield
```

The model uses packed sequences so that samples with different numbers of growth-stage observations can be processed efficiently.

---

## 5. Hyperparameter Optimization

LSTM hyperparameters are optimized using **Optuna**.

The optimization process considers:

| Hyperparameter          | Description                                          |
| ----------------------- | ---------------------------------------------------- |
| Hidden size             | Number of LSTM hidden units                          |
| Number of layers        | Number of stacked LSTM layers                        |
| Dropout rate            | Dropout applied within the model                     |
| Batch size              | Number of samples per training batch                 |
| Learning rate           | Adam optimizer learning rate                         |
| Early-stopping patience | Number of epochs without improvement before stopping |

The optimization objective is based on prediction error evaluated through cross-validation.

---

## 6. Model Evaluation

Model performance is evaluated using three commonly used regression metrics:

### R² — Coefficient of Determination

Measures the proportion of variance in observed maize yield explained by the model.

### RMSE — Root Mean Square Error

Measures the magnitude of prediction errors while giving greater weight to larger errors.

### MAE — Mean Absolute Error

Measures the average absolute difference between observed and predicted yield.

The final model is evaluated using an **independent test dataset that is not used for model optimization**.

---

## 7. Cross-Validation

Five-fold cross-validation is used during model and feature-set evaluation.

Cross-validation is used to:

* assess model stability,
* compare feature-group combinations,
* reduce dependence on a single training/validation split, and
* support selection of the most suitable feature configuration.

The independent test dataset is reserved for the final evaluation.

---

## 8. Data Organization

The Excel datasets should be placed directly inside the `data/` directory.

```text
maize-yield-lstm/
│
├── data/
│   ├── dataset_1.xlsx
│   ├── dataset_2.xlsx
│   └── ...
│
├── main.py
├── config/
├── src/
└── outputs/
```

Each Excel workbook is expected to contain the sheets required by the modeling workflow, including the training and testing datasets.

### Data availability

The experimental datasets are not currently included in this repository because the associated research is under publication/review.

Following publication, data availability will be considered subject to:

* institutional requirements,
* project or funding restrictions,
* data ownership,
* participant or field-data agreements, where applicable, and
* journal data-sharing policies.

If the data cannot be publicly released, the repository will continue to provide the complete computational workflow required to reproduce the analysis using an appropriately formatted dataset.

---

## 9. Repository Structure

The repository is organized into modular Python files so that individual components of the workflow can be inspected, reused, and maintained independently.

```text
maize-yield-lstm/
│
├── main.py                         # Main entry point
│
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git exclusions
├── LICENSE                         # Software license
│
├── config/
│   ├── __init__.py
│   └── settings.py                 # Paths and configuration
│
├── src/
│   ├── __init__.py
│   │
│   ├── data_loader.py              # Excel data loading
│   ├── preprocessing.py            # Preprocessing and sequences
│   ├── feature_selection.py        # Correlation-based selection
│   ├── dataset.py                  # PyTorch Dataset
│   ├── models.py                   # LSTM architecture
│   ├── training.py                 # Model training and early stopping
│   ├── optimization.py             # Optuna optimization
│   ├── evaluation.py               # RMSE, R², and MAE
│   ├── experiment.py               # Complete experiment workflow
│   └── utils.py                    # Reproducibility and utilities
│
├── data/
│   ├── README.md                   # Data documentation
│   └── *.xlsx                      # Experimental data
│
└── outputs/
    ├── results/                    # Model and experiment results
    ├── models/                     # Saved trained models
    └── logs/                       # Experiment logs
```

---

## 10. Running the Model

### Step 1 — Clone the repository

```bash
git clone <repository-url>
cd maize-yield-lstm
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Add the data

Place the required Excel files directly in:

```text
data/
```

For example:

```text
data/
├── maize_dataset.xlsx
└── ...
```

### Step 4 — Run the experiment

From the project root directory:

```bash
python main.py
```

The complete workflow will then be executed through `src/experiment.py`.

---

## 11. Configuration

Project-level settings are maintained in:

```text
config/settings.py
```

These settings include:

* project and data paths,
* output directories,
* random seed,
* computational device, and
* other experiment-level configuration parameters.

The project is designed to automatically use a CUDA-enabled GPU when available.

---

## 12. Outputs

Results generated by the workflow are organized under:

```text
outputs/
├── results/
├── models/
└── logs/
```

Depending on the experiment configuration, the results may include:

* feature-selection results,
* cross-validation performance,
* Optuna trial results,
* optimized hyperparameters,
* final model performance,
* observed versus predicted yield values,
* trained LSTM model files, and
* processing-time information.

---

## 13. Reproducibility

A fixed random seed is used to improve experiment reproducibility.

The repository separates:

* data loading,
* preprocessing,
* feature selection,
* model definition,
* training,
* hyperparameter optimization, and
* evaluation.

This modular organization allows individual components to be independently inspected and reused.

For publication-quality reproduction, the following should be kept consistent:

* dataset version,
* training/testing partition,
* feature definitions,
* preprocessing procedure,
* random seed,
* software versions,
* PyTorch version,
* Optuna version, and
* hardware configuration where relevant.

---

## 14. Research Context

This repository supports research on **growth stage-specific maize yield prediction using UAV multispectral and multi-source data with explainable/deep learning approaches**.

The framework is intended to support investigation of how information collected at different stages of crop development contributes to maize yield prediction.

The code is structured to facilitate future extensions, including:

* additional remote sensing variables,
* additional environmental variables,
* alternative machine-learning models,
* explainable AI methods,
* spatial prediction, and
* evaluation across additional fields and growing seasons.

---

## 15. Citation

If you use this code or methodology in academic research, please cite the associated publication when it becomes available.

A DOI and formal citation information will be added to this section following publication.

---

## 16. License

This project is distributed under the license specified in:

```text
LICENSE
```

Please review the license before reusing or redistributing the code.

---

## 17. Contact

For questions regarding the modeling framework, implementation, or research collaboration, please refer to the contact information associated with the project or its corresponding publication.
