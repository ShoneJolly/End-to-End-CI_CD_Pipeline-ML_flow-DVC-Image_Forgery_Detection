import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageChops, ImageEnhance
import os
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionPipeline:
    def __init__(self):
        try:
            model_path = os.path.join("models", "model.keras")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at: {model_path}")
            self.model = load_model(model_path)
            logger.info("Model loaded successfully")
            logger.info(f"Model input shape: {self.model.input_shape}")
            logger.info(f"Model output shape: {self.model.output_shape}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise Exception(f"Failed to load model: {str(e)}")

    def img_difference(self, org_img):
        """Compute the enhanced difference between original and resaved images."""
        try:
            # Resave the image as JPEG in memory
            img_io = io.BytesIO()
            org_img.save(img_io, 'JPEG', quality=90, optimize=True)
            resaved_img = Image.open(img_io).convert('RGB')
            
            # Compute difference
            diff = ImageChops.difference(org_img, resaved_img)
            extrema = diff.getextrema()
            max_diff = max([ex[1] for ex in extrema])
            if max_diff == 0:
                max_diff = 1
            scale = 255.0 / max_diff  # Adjust if normalization_scale differs
            diff = ImageEnhance.Brightness(diff).enhance(scale)
            diff = ImageEnhance.Sharpness(diff).enhance(2.0)  # Adjust sharpness_factor if needed
            logger.debug("Difference image computed")
            return diff
        except Exception as e:
            logger.error(f"Error computing difference image: {str(e)}")
            raise Exception(f"Error computing difference image: {str(e)}")

    def preprocess_image(self, img):
        """Preprocess the in-memory PIL image for prediction."""
        if not isinstance(img, Image.Image):
            raise TypeError(f"Expected PIL.Image object, got {type(img)}")
        try:
            # Compute difference image
            diff_img = self.img_difference(img)
            # Resize and normalize
            diff_img = diff_img.resize((128, 128))
            img_array = np.array(diff_img, dtype=np.float32) / 255.0  # Adjust if normalization_scale differs
            img_array = img_array.reshape(-1, 128, 128, 3)
            logger.info(f"Preprocessed image shape: {img_array.shape}")
            logger.info(f"Preprocessed image min: {img_array.min()}, max: {img_array.max()}")
            return img_array
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise Exception(f"Preprocessing failed: {str(e)}")

    def predict(self, img):
        """Run prediction on the in-memory PIL image."""
        try:
            img_array = self.preprocess_image(img)
            pred = self.model.predict(img_array)[0]
            logger.info(f"Raw model output: {pred}")
            if len(pred) != 2:
                logger.error(f"Unexpected output shape: {pred.shape}. Expected 2 classes.")
                raise ValueError(f"Model output has {len(pred)} values, expected 2.")
            # Class order: pred[0] = Not Forged (Au), pred[1] = Forged (Tp)
            # Convert confidence to percentage with 2 decimal places
            confidence = pred[0] * 100 if pred[0] > pred[1] else pred[1] * 100
            return f"Not Forged (confidence: {confidence:.2f}%)" if pred[0] > pred[1] else f"Forged (confidence: {confidence:.2f}%)"
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise Exception(f"Prediction failed: {str(e)}")

    def get_image_details(self, img, file_size_bytes):
        """Extract image details from the in-memory PIL image."""
        if not isinstance(img, Image.Image):
            raise TypeError(f"Expected PIL.Image object, got {type(img)}")
        return {
            "format": img.format if img.format else "Unknown",
            "size": f"{img.size[0]} x {img.size[1]} pixels",
            "file_size": f"{file_size_bytes / 1024:.2f} KB"
        }