# Dental Health Detection Flask Deployment

This project is a web application for dental health detection using an ONNX object detection model. The application allows users to upload an image and receive a prediction with bounding boxes around detected dental issues.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Acknowledgments](#acknowledgments)

## Requirements

- Python 3.6 or higher
- Flask
- OpenCV
- NumPy
- ONNX Runtime

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/sayyidan-i/flask-dental-health-detection.git
   cd flask-dental-health-detection
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Ensure you have your ONNX model file in the `model` directory. Update the `model_onnx` variable in `main.py` if your model has a different name or path.

## Usage

1. Start the Flask application:

   ```bash
   flask run
   ```

2. Open your web browser and go to `http://127.0.0.1:5000`.

3. Upload an image and click the "Predict" button. The result image with bounding boxes will be displayed on the web page.

## Project Structure

```
dental-health-detection/
├── model/
│   └── FastestDet_352_AP4.5.onnx  # Place your ONNX model here
├── uploads/                        # Directory where result images are saved
├── templates/
│   └── index.html                  # HTML template for the web app
├── main.py                         # Main application file
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## Acknowledgments

This project uses the following libraries and frameworks:

- [Flask](https://flask.palletsprojects.com/)
- [OpenCV](https://opencv.org/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [FastestDet](https://github.com/dog-qiuqiu/FastestDet)


Special thanks to the contributors of these projects for their amazing work.