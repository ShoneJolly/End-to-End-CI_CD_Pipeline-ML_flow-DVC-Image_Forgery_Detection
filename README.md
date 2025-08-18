# End-to-End Image Forgery Detection with MLflow, DVC, and AWS CI/CD

## Objective

The goal of this project is to develop, track, and deploy an Image Forgery Detection model using the CASIA2 dataset. The model is built using a custom CNN architecture for high-accuracy feature extraction and classification. The entire workflow integrates MLflow for experiment tracking, DVC for data and pipeline management, and automated CI/CD to AWS using Docker and GitHub Actions.


<img width="1100" height="720" alt="Image Forgery Detection-CICD Pipeline Architecture" src="https://github.com/user-attachments/assets/d378414d-09a0-468f-a649-7be4a67b1abe" />


---
## 1. Project Setup and Configuration

- Organized the project with:
    - `config.yaml` for all paths and global settings
    - `params.yaml` for hyperparameters of preprocessing and training
- Created entity classes for structured configuration handling  
- Implemented a Configuration Manager in src/config to load settings  
- Built components for:
  - Data ingestion
  - Data preprocessing
  - Model training
  - Model evaluation
- Linked components into pipelines for automation  
- Added `main.py` to trigger pipelines and updated `dvc.yaml` to define the DVC pipeline stages

---

## 2. Model Development

- Created a custom CNN architecture for image forgery detection using the CASIA2 dataset  
- Evaluated the model using the following metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 score

---

## 3. Experiment Tracking with MLflow

- Integrated MLflow to:
  - Log parameters and performance metrics  
  - Store trained image forgery detection model artifacts  
  - Tag model versions
- Configured DagsHub as the remote MLflow tracking server  
- Used mlflow ui to visualize and compare experiments  
- Set environment variables to push experiment logs to the remote server

---

## 4. Data Version Control with DVC

- Initialized DVC to manage datasets and pipeline stages  
- Ensured reproducibility by versioning all data and models  
- Used:
  - dvc repro to rerun the pipeline from any stage  
  - dvc dag to visualize the workflow
-Simplified collaboration and maintained a clear experiment history

---

## 5. Deployment with AWS and GitHub Actions

- Used an **automated CI/CD pipeline** with GitHub Actions and AWS services  
- Configured an **EC2 instance** as a self-hosted GitHub Actions runner  
- During the pipeline:
  - Builds the Docker image for the model inference service  
  - Pushes the image to AWS **ECR**  
- EC2 pulls the image from ECR and runs it as a container to serve predictions
- Applied IAM policies:
  - AmazonEC2FullAccess
  - AmazonEC2ContainerRegistryFullAccess
- Stored AWS credentials and configuration details as GitHub Secrets

---

## 6. Web Application for Image Forgery Detection

- Developed a **Flask-based web application** for user interaction  
- Allows uploading of images to predict whether an image is forged or not forged  
- Uses the trained image forgery detection model for predictions via a REST API and HTML front-end  
- Runs locally using `app.py` during development  
- Configured to listen on a custom port in AWS after CI/CD deployment

---
<img width="1100" height="720" alt="Screenshot 2025-08-11 150655" src="https://github.com/user-attachments/assets/b92f5223-61d4-4e80-b516-7a2939dad7c6" />

## Outcome

- Production-ready image forgery detection system using a CNN architecture  
- Automated training and evaluation pipelines with DVC  
- Experiment tracking and model versioning using MLflow  
- Fully automated deployment to AWS using Docker and GitHub Actions  
- User-friendly web application for real-time image forgery detection from PNG, JPEG, and JPG format images

---

## Key Tools and Technologies

- Python (data handling and model training)  
- TensorFlow/Keras (CNN architecture model)  
- MLflow (experiment tracking)  
- DVC (data and pipeline version control)  
- Docker (containerization)  
- AWS ECR and EC2 (cloud deployment)  
- GitHub Actions (CI/CD automation)  
- DagsHub (remote MLflow server)  
- Flask (web application framework)


