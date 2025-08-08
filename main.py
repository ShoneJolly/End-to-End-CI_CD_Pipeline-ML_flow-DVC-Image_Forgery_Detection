from ImageForgeryDetection import logger
from ImageForgeryDetection.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from ImageForgeryDetection.pipeline.stage_02_data_preprocessing import DataPreprocessingTrainingPipeline
from ImageForgeryDetection.pipeline.stage_03_model_trainer import ModelTrainingPipeline

STAGE_NAME = "Data Ingestion stage"

try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx=================x")
except Exception as e:
        logger.exception(e)
        raise e
    
STAGE_NAME = "Data Preprocessing stage"

try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = DataPreprocessingTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx=================x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Model Training stage"

try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx=================x")
except Exception as e:
        logger.exception(e)
        raise e