from ImageForgeryDetection.config.configuration import ConfigurationManager
from ImageForgeryDetection.components.model_evaluation import ModelEvaluation
from ImageForgeryDetection import logger

STAGE_NAME = "Model Evaluation stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_model_evaluation_config()
        evaluation = ModelEvaluation(eval_config)
        evaluation.evaluation()
        # evaluation.log_into_mlflow()
        logger.info("Model evaluation pipeline completed")


if __name__=='__main__':
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx=================x")
    except Exception as e:
        logger.exception(e)
        raise e
    
