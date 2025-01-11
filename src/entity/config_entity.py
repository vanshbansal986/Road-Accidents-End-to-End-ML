from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir: Path
    curr_data_path: Path
    store_data_path: Path 


@dataclass
class DataCleaningConfig:
    root_dir: Path
    curr_file_path: Path
    clean_file_path: Path
    train_file_path: Path
    test_file_path: Path
    main_data_dir: Path


@dataclass
class PreprocessingConfig:
    root_dir: Path
    preprocesser_obj: Path
    train_file_path: Path
    test_file_path: Path
    preprocessed_x_train: Path
    preprocessed_x_test: Path
    preprocessed_y_train: Path
    preprocessed_y_test: Path

@dataclass
class ModelTrainerConfig:
    root_dir : Path
    results : Path
    preprocessed_x_train : Path
    preprocessed_x_test : Path
    preprocessed_y_train : Path
    preprocessed_y_test : Path