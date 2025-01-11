from src.components.preprocessing import Preprocessing
from src.config.configuration import ConfigurationManager

class PreprocessingPipeline:
    def __init__(self):
        pass

    def initiate_preprocessing(self):
        try:
            config = ConfigurationManager()
            preprocessing_obj = config.get_preprocessing_config()
            preprocessing = Preprocessing(config=preprocessing_obj)
            preprocessing.start_preprocessing()
        except Exception as e:
            raise e