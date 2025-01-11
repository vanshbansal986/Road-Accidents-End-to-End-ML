# Road Accident Analysis and Prediction - End-to-End Machine Learning Project

## Table of Contents
- [Road Accident Analysis and Prediction - End-to-End Machine Learning Project](#road-accident-analysis-and-prediction---end-to-end-machine-learning-project)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
    - [Key Objectives](#key-objectives)
    - [Key Components Used](#key-components-used)
  - [Project Structure](#project-structure)
- [Project Overview](#project-overview)
  - [Project Structure](#project-structure-1)
  - [Pipeline Components](#pipeline-components)
    - [1. Data Ingestion](#1-data-ingestion)
    - [2. Data Cleaning](#2-data-cleaning)
    - [3. Data Preprocessing](#3-data-preprocessing)
    - [4. Model Training \& Evaluation](#4-model-training--evaluation)
    - [5. Model Deployment](#5-model-deployment)
  - [Setup](#setup)
  - [Usage](#usage)
  - [License](#license)

## Overview
This project focuses on analyzing and predicting road accidents using machine learning models. The pipeline involves data ingestion, validation, transformation, model training, evaluation, and deployment. 

### Key Objectives
- Analyze patterns and contributing factors to road accidents.
- Predict the severity or likelihood of road accidents using historical data.

### Key Components Used
- **Database**: SQLite
- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Deployment**: Flask (or any preferred framework)

## Project Structure
# Project Overview

This project aims to clean, process, and train a machine learning model using data from US accidents (March 2023). It is structured to allow seamless ingestion, preprocessing, and training of models, while also enabling detailed logging and result tracking.

## Project Structure

```
├── Research
│   ├── data_cleaning.ipynb          # Jupyter Notebook for data cleaning
│   ├── data_ingestion.ipynb        # Jupyter Notebook for data ingestion
│   ├── model_trainer_.ipynb        # Jupyter Notebook for model training
│   └── preprocessor.ipynb          # Jupyter Notebook for preprocessing
├── US_Accidents_March23 copy.csv    # Raw dataset
├── artifacts                       # Folder for storing processed data, model results, and other artifacts
│   ├── data_cleaning
│   │   └── clean_data.csv          # Cleaned dataset
│   ├── data_ingestion
│   │   └── data.csv                # Ingested raw data
│   ├── main_data
│   │   ├── test_data.csv           # Processed test data
│   │   └── train_data.csv          # Processed train data
│   ├── model_trainer
│   │   └── results.txt             # Model evaluation results
│   └── preprocessing
│       ├── preprocesser.pkl        # Preprocessing model saved as pickle
│       ├── x_test_trf.npy          # Processed test features
│       ├── x_train_trf.npy         # Processed train features
│       ├── y_test_trf.npy          # Processed test labels
│       └── y_train_trf.npy         # Processed train labels
├── config.yaml                     # Configuration file for model training and processing
├── data.csv                        # Original dataset in CSV format
├── data.html                       # HTML representation of the data
├── logs                            # Folder for logs
│   └── logging.log                 # Logs for training and processing activities
├── main.py                         # Main entry point for the project
├── readme.md                       # This README file
├── requirements.txt                # Python package dependencies
├── sampled_data.csv                # Sampled subset of the dataset
├── src                             # Source code for various modules
│   ├── __init__.py
│   ├── __pycache__                 # Cached bytecode files
│   ├── components                  # Folder for specific components like data cleaning, ingestion, etc.
│   │   ├── __init__.py
│   │   ├── data_cleaning.py        # Code for data cleaning operations
│   │   ├── data_ingestion.py      # Code for data ingestion
│   │   ├── model_trainer.py       # Code for model training
│   │   └── preprocessing.py       # Code for preprocessing operations
│   ├── config                      # Folder for configuration-related code
│   │   ├── __init__.py
│   │   └── configuration.py       # Code for handling configuration settings
│   ├── constants                   # Folder for constants used across the project
│   │   ├── __init__.py
│   ├── entity                      # Folder for entity-related code
│   │   ├── config_entity.py       # Code for entity configurations
│   ├── pipeline                    # Folder for pipeline-related code
│   │   ├── data_cleaning.py       # Pipeline code for data cleaning
│   │   ├── data_ingestion.py      # Pipeline code for data ingestion
│   │   ├── model_trainer.py       # Pipeline code for model training
│   │   └── preprocessing.py       # Pipeline code for preprocessing
│   └── utils                       # Utility functions
│       ├── __init__.py
│       └── common.py              # Common helper functions
└── test.ipynb                      # Jupyter Notebook for testing
```

## Pipeline Components
### 1. Data Ingestion
- Collect data from CSV file.
- Save raw data into the `artifacts/data_ingestion` folder.
- Log metadata for reproducibility.

### 2. Data Cleaning
The data cleaning process involved the following steps:

1. **Handling Date-Time Variables**: 
   - Converted `Start_Time` and `End_Time` to datetime format.
   - Extracted features like year, month, day, hour, weekday, and weekend indicator from `Start_Time`.

2. **Dropping Unnecessary Columns**: 
   - Removed irrelevant columns such as `ID`, `Description`, `City`, and other location-specific or redundant fields.

3. **Outlier Treatment**: 
   - Used the Interquartile Range (IQR) method to handle outliers in numerical columns, ensuring they fall within acceptable bounds.

4. **Feature Categorization**: 
   - Segregated the dataset into numerical and categorical features for further processing.

### 3. Data Preprocessing
The preprocessing stage focused on preparing the cleaned data for modeling by applying a series of transformations:

1. **Handling Missing Values**:
   - Imputed missing values in numerical columns using the median strategy.
   - Imputed missing values in categorical columns using the most frequent value, followed by one-hot encoding.

2. **Feature Transformation**:
   - Applied **PowerTransformer** to normalize numerical features and stabilize variance.
   - Standardized numerical features using **StandardScaler** for uniform scaling.

3. **Dimensionality Reduction**:
   - Reduced feature dimensionality using **Principal Component Analysis (PCA)**, retaining 15 components to optimize performance.

4. **Pipeline Implementation**:
   - Combined all preprocessing steps into a unified pipeline:
     - Imputation and encoding of categorical and numerical columns.
     - Normalization, scaling, and PCA transformation.
   - Ensured seamless transformation for both training and testing datasets.

5. **Label Encoding**:
   - Encoded the target variable (`Severity`) using **LabelEncoder** for compatibility with machine learning models.

6. **Saving Preprocessed Data**:
   - Saved transformed features (`x_train`, `x_test`) and target labels (`y_train`, `y_test`) as `.npy` files.
   - Persisted the preprocessing pipeline as a joblib object for reuse during inference.

These preprocessing steps ensure that the data is clean, transformed, and ready for machine learning models while maintaining consistency between training and testing phases.
This provides a concise yet clear explanation of the preprocessing workflow. Let me know if you need further edits!








### 4. Model Training & Evaluation


This script provides functionality for training and evaluating machine learning models using various classifiers, such as Logistic Regression, K-Nearest Neighbors (KNN), Random Forest (RF), Support Vector Machine (SVM), XGBoost, and Neural Networks.


1. **Initialization**: 
   - The `ModelTrainer` class is initialized with a configuration object (`ModelTrainerConfig`) that contains the paths to the preprocessed datasets and the results file where metrics will be saved.

2. **Log Metrics**:
   - The `log_metrics` function calculates and logs key evaluation metrics: Accuracy, Precision, Recall, and F1 Score, for each model to a specified results file.

3. **Model Training**:
   - The `start_model_training` function loads the training and test datasets and trains the models:
     - **Logistic Regression**
     - **K-Nearest Neighbors (KNN)**
     - **Random Forest (RF)**
     - **Support Vector Machine (SVM)**
     - **XGBoost**
     - **Neural Networks (NN)**

4. **Model Evaluation**:
   - After training each model, the function logs the performance metrics in the results file.



### 5. Model Deployment
- Deploy the trained model as a REST API using Flask or a similar framework.
- Provide endpoints for prediction and model monitoring.

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/username/road_accident_project.git
   cd road_accident_project
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up the database:
   - Create the SQLite database.
   - Populate it with raw data from the `data/` folder.

4. Run the pipeline:
   ```bash
   python src/pipeline.py
   ```

## Usage
- For exploratory analysis, use the Jupyter notebooks in the `notebooks/` folder.
- To run the API locally:
  ```bash
  python src/app.py
  ```
- Access the API at `http://127.0.0.1:5000/predict`.

## License
This project is licensed under the [MIT License](LICENSE).
