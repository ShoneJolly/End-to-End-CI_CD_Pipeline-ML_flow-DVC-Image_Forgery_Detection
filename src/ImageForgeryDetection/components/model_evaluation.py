from ImageForgeryDetection import logger
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from pathlib import Path
from ImageForgeryDetection.utils.common import save_json
import os
import tempfile
from ImageForgeryDetection.entity.config_entity import ModelEvaluationConfig

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.model = None
        self.X_test = None
        self.y_test = None
        self.score = None

    def load_data(self):
        """Loads test data from joblib files specified in config."""
        logger.info(f"Loading test data from {self.config.load_data}")
        try:
            x_path = Path(self.config.load_data) / 'X_90.joblib'
            y_path = Path(self.config.load_data) / 'y.joblib'
            X = joblib.load(x_path)
            y = joblib.load(y_path)
            logger.info(f"Loaded X with shape {X.shape} and y with shape {y.shape}")
            return X, y
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def split_data(self, X, y):
        """Splits data into training and testing sets."""
        logger.info("Splitting data into train and test sets")
        try:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            logger.info(f"Test set: X_test shape {X_test.shape}, y_test shape {y_test.shape}")
            return X_test, y_test
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            raise

    def preprocess_data(self, X_test, y_test):
        """Reshapes test data for CNN input."""
        logger.info("Preprocessing test data")
        try:
            X_test = X_test.reshape(X_test.shape[0], 128, 128, 3)
            y_test = y_test.reshape(y_test.shape[0], 2)
            logger.info(f"Reshaped X_test to {X_test.shape}, y_test to {y_test.shape}")
            self.X_test, self.y_test = X_test, y_test
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        """Loads the trained model."""
        logger.info(f"Loading model from {path}")
        try:
            return tf.keras.models.load_model(path)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def get_test_generator(self):
        """Returns a Sequence generator for test data."""
        class TestGenerator(Sequence):
            def __init__(self, X, y, batch_size, **kwargs):
                super().__init__(**kwargs)  # Initialize Sequence base class
                self.X = X
                self.y = y
                self.batch_size = batch_size
                self.indexes = np.arange(len(self.X))

            def __len__(self):
                return int(np.floor(len(self.X) / self.batch_size))

            def __getitem__(self, index):
                indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
                X = [self.X[k] for k in indexes]
                y = [self.y[k] for k in indexes]
                return np.array(X), np.array(y)

        return TestGenerator(self.X_test, self.y_test, self.config.params['batch_size'])

    def evaluation(self):
        """Evaluates the model and saves scores."""
        logger.info("Starting model evaluation")
        try:
            # Load and preprocess data
            X, y = self.load_data()
            X_test, y_test = self.split_data(X, y)
            self.preprocess_data(X_test, y_test)

            # Load model
            model_path = Path(self.config.model_path) / self.config.model
            self.model = self.load_model(model_path)

            # Create test generator
            test_generator = self.get_test_generator()

            # Evaluate model
            logger.info("Evaluating model on test data")
            self.score = self.model.evaluate(
                test_generator,
                batch_size=self.config.params['batch_size'],
                return_dict=True
            )
            logger.info(f"Evaluation scores: {self.score}")

            # Save scores
            self.save_score()

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise

    def save_score(self):
        """Saves evaluation scores to a JSON file."""
        logger.info("Saving evaluation scores")
        try:
            # Handle F1 score as a tensor array by computing mean
            f1_score = self.score.get('f1_score', 0.0)
            if isinstance(f1_score, tf.Tensor):
                f1_score = np.mean(f1_score.numpy())
            elif isinstance(f1_score, np.ndarray):
                f1_score = np.mean(f1_score)

            scores = {
                "loss": float(self.score.get('loss', 0.0)),
                "accuracy": float(self.score.get('accuracy', 0.0)),
                "precision": float(self.score.get('precision', 0.0)),
                "recall": float(self.score.get('recall', 0.0)),
                "f1_score": float(f1_score)
            }
            save_json(path=Path("scores.json"), data=scores)
            logger.info("Scores saved successfully")
        except Exception as e:
            logger.error(f"Error saving scores: {e}")
            raise

    def log_into_mlflow(self):
        """Logs parameters, metrics, and model to MLflow."""
        logger.info("Logging to MLflow")
        try:
            mlflow.set_tracking_uri(self.config.mlflow_uri)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            logger.info(f"MLflow tracking URI: {self.config.mlflow_uri}")

            with mlflow.start_run():
                # Log parameters
                mlflow.log_params(self.config.params)

                # Handle F1 score for logging
                f1_score = self.score.get('f1_score', 0.0)
                if isinstance(f1_score, tf.Tensor):
                    f1_score = np.mean(f1_score.numpy())
                elif isinstance(f1_score, np.ndarray):
                    f1_score = np.mean(f1_score)

                # Log metrics
                mlflow.log_metrics({
                    "loss": float(self.score.get('loss', 0.0)),
                    "accuracy": float(self.score.get('accuracy', 0.0)),
                    "precision": float(self.score.get('precision', 0.0)),
                    "recall": float(self.score.get('recall', 0.0)),
                    "f1_score": float(f1_score)
                })

                # Save model to a temporary .keras file
                with tempfile.TemporaryDirectory() as tmpdirname:
                    temp_model_path = os.path.join(tmpdirname, "model.keras")
                    logger.info(f"Saving model to temporary path: {temp_model_path}")
                    self.model.save(temp_model_path)
                    if not os.path.exists(temp_model_path):
                        raise FileNotFoundError(f"Failed to save model at {temp_model_path}")
                    logger.info(f"Verified model saved at {temp_model_path}, size: {os.path.getsize(temp_model_path)} bytes")

                    # Log model as artifact
                    logger.info("Logging model as MLflow artifact")
                    mlflow.log_artifact(temp_model_path, artifact_path="model")
                    logger.info("Model logged to MLflow as artifact successfully")

                    # Register the model in MLflow Model Registry
                    if tracking_url_type_store != "file":
                        logger.info("Registering model in MLflow as ImageForgeryDetectionModel")
                        client = mlflow.tracking.MlflowClient()
                        run_id = mlflow.active_run().info.run_id
                        try:
                            # Create or update model in registry
                            result = client.create_model_version(
                                name="ImageForgeryDetectionModel",
                                source=f"{mlflow.get_artifact_uri('model')}",
                                run_id=run_id
                            )
                            logger.info(f"Model registered as ImageForgeryDetectionModel, version {result.version}")
                        except mlflow.exceptions.RestException as e:
                            logger.error(f"Failed to register model: {e.json}, Status Code: {e.status_code}")
                            raise
                        except Exception as e:
                            logger.error(f"Unexpected error during model registration: {str(e)}")
                            raise

        except Exception as e:
            logger.error(f"Error logging to MLflow: {e}")
            raise