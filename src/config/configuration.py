from src.utils.common import read_yaml , create_directories
from src.constants import CONFIG_FILE_PATH
from src.entity.config_entity import DataIngestionConfig , DataCleaningConfig , PreprocessingConfig , ModelTrainerConfig

class ConfigurationManager:
    def __init__(self ,config_filepath=CONFIG_FILE_PATH):
        
        self.config = read_yaml(config_filepath)

        create_directories([self.config.artifacts_root])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            curr_data_path = config.curr_data_path,
            store_data_path = config.store_data_path
        )

        return data_ingestion_config

    def get_data_cleaning_config(self) -> DataCleaningConfig:
        config = self.config.data_cleaning
        create_directories([config.root_dir])
        create_directories([config.main_data_dir])

        data_cleaning_config = DataCleaningConfig(
            root_dir = config.root_dir,
            curr_file_path = config.curr_file_path,
            clean_file_path = config.clean_file_path,
            train_file_path = config.train_file_path,
            test_file_path = config.test_file_path,
            main_data_dir = config.main_data_dir
        )

        return data_cleaning_config
    
    def get_preprocessing_config(self) -> PreprocessingConfig:
        config = self.config.preprocessing
        create_directories([config.root_dir])
        

        preprocessing_config = PreprocessingConfig(
            root_dir = config.root_dir,
            preprocesser_obj = config.preprocesser_obj,
            train_file_path = config.train_file_path,
            test_file_path = config.test_file_path,
            preprocessed_x_train = config.preprocessed_x_train,
            preprocessed_x_test = config.preprocessed_x_test,
            preprocessed_y_train = config.preprocessed_y_train,
            preprocessed_y_test = config.preprocessed_y_test
        )

        return preprocessing_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir = config.root_dir,
            results = config.results,
            preprocessed_x_train = config.preprocessed_x_train,
            preprocessed_x_test = config.preprocessed_x_test,
            preprocessed_y_train = config.preprocessed_y_train,
            preprocessed_y_test = config.preprocessed_y_test,
        )

        return model_trainer_config