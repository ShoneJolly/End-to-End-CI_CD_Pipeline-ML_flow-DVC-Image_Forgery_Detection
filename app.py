from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from PIL import Image
import io
import base64
from ImageForgeryDetection.pipeline.prediction import PredictionPipeline

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.forgerydetection = PredictionPipeline()

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Validate file type
        if not file.mimetype in ['image/png', 'image/jpeg', 'image/jpg']:
            return jsonify({"error": "Unsupported file type. Please upload PNG or JPEG."}), 400
        
        # Read the image in memory
        img_data = file.read()
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        
        # Get prediction and image details
        result = clApp.forgerydetection.predict(img)
        details = clApp.forgerydetection.get_image_details(img, len(img_data))
        
        # Convert image to base64 for display (in-memory)
        img_io = io.BytesIO()
        img.save(img_io, format='JPEG')
        image_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
        
        # Prepare response
        response = {
            "prediction": result,
            "format": details["format"],
            "size": details["size"],
            "file_size": details["file_size"],
            "image_base64": image_base64
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": f"Error in predictRoute: {str(e)}"}), 500

if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=8080)