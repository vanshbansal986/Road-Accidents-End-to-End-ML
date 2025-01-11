from src.components.model_trainer import ModelTrainer
from src.config.configuration import ConfigurationManager

class ModelTrainerPipeline:
    def __init__(self):
        pass

    def initiate_model_training(self):
        try:
            config = ConfigurationManager()
            model_training_obj = config.get_model_trainer_config()
            model_training = ModelTrainer(config=model_training_obj)
            model_training.start_model_training()
        except Exception as e:
            raise e