import os
import zipfile
import gdown
from ImageForgeryDetection import logger
from ImageForgeryDetection.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        logger.info("Initializing DataIngestion")
        self.config = config


    
     
    def download_file(self)-> str:
        '''
        Fetch data from the url
        '''

        try: 
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+file_id,zip_download_dir)

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            raise e
        
    
    def extract_zip_file(self):
        """
        Extracts the zip file into the data directory and deletes it
        """
        unzip_path = self.config.unzip_dir
        zip_file_path = self.config.local_data_file
        logger.info(f"Extracting {zip_file_path} to {unzip_path}")
        try:
            os.makedirs(unzip_path, exist_ok=True)
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
            logger.info(f"Extracted to {unzip_path}")
            logger.info(f"Deleting {zip_file_path}")
            os.remove(zip_file_path)
            logger.info(f"Deleted {zip_file_path}")
        except Exception as e:
            logger.error(f"Error extracting/deleting {zip_file_path}: {e}")
            raise