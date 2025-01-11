from src import logger
from src.pipeline.data_ingestion import DataIngestionPipeline
from src.pipeline.data_cleaning import DataCleaningPipeline
from src.pipeline.preprocessing import PreprocessingPipeline
from src.pipeline.model_trainer import ModelTrainerPipeline

STAGE_NAME = "Data Ingestion"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionPipeline()
    obj.initiate_data_ingestion()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Cleaning"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataCleaningPipeline()
    obj.initiate_data_cleaning()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "PreProcessing"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = PreprocessingPipeline()
    obj.initiate_preprocessing()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Training"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = ModelTrainerPipeline()
    obj.initiate_model_training()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e