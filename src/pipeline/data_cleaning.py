from src.components.data_cleaning import DataCleaning
from src.config.configuration import ConfigurationManager

class DataCleaningPipeline:
    def __init__(self):
        pass

    def initiate_data_cleaning(self):
        try:
            config = ConfigurationManager()
            data_cleaning_config = config.get_data_cleaning_config()
            data_cleaning = DataCleaning(config=data_cleaning_config)
            data_cleaning.start_data_cleaning()
        except Exception as e:
            raise e