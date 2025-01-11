from src.config.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion

class DataIngestionPipeline:
    def __init__(self):
        pass

    def initiate_data_ingestion(self):
        try:
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.store_data()
        except Exception as e:
            raise e