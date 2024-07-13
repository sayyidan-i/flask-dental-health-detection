import json
import numpy as np
from flask import Flask, request, render_template, jsonify, send_from_directory
import cv2
import onnxruntime
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load ONNX model
model_onnx = "model/FastestDet_352_AP4.5.onnx"
session = onnxruntime.InferenceSession(model_onnx)

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def tanh(x):
    return 2. / (1 + np.exp(-2 * x)) - 1

def preprocess(src_img, size):
    output = cv2.resize(src_img, (size[0], size[1]), interpolation=cv2.INTER_AREA)
    output = output.transpose(2, 0, 1)
    output = output.reshape((1, 3, size[1], size[0])) / 255
    return output.astype('float32')

def nms(dets, thresh=0.45):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]

        order = order[inds + 1]

    output = []
    for i in keep:
        output.append(dets[i].tolist())

    return output

def detection(session, img, input_width, input_height, thresh):
    pred = []
    H, W, _ = img.shape
    data = preprocess(img, [input_width, input_height])
    input_name = session.get_inputs()[0].name
    feature_map = session.run([], {input_name: data})[0][0]
    feature_map = feature_map.transpose(1, 2, 0)
    feature_map_height = feature_map.shape[0]
    feature_map_width = feature_map.shape[1]

    for h in range(feature_map_height):
        for w in range(feature_map_width):
            data = feature_map[h][w]

            obj_score, cls_score = data[0], data[5:].max()
            score = (obj_score ** 0.6) * (cls_score ** 0.4)

            if score > thresh:
                cls_index = np.argmax(data[5:])
                x_offset, y_offset = tanh(data[1]), tanh(data[2])
                box_width, box_height = sigmoid(data[3]), sigmoid(data[4])
                box_cx = (w + x_offset) / feature_map_width
                box_cy = (h + y_offset) / feature_map_height
                
                # cx, cy, w, h => x1, y1, x2, y2
                x1, y1 = box_cx - 0.5 * box_width, box_cy - 0.5 * box_height
                x2, y2 = box_cx + 0.5 * box_width, box_cy + 0.5 * box_height
                x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)

                pred.append([x1, y1, x2, y2, score, cls_index])

    return nms(np.array(pred))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def predict():
    file = request.files['file']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    input_width = 352
    input_height = 352
    thresh = 0.5
    bboxes = detection(session, img, input_width, input_height, thresh)
    
    # Load label names
    names = ["Normal", "Karies kecil", "Karies sedang", "Karies besar", "Stain", "Karang gigi", "Lain-Lain"]
            
    # Initialize colors for each label
    label_colors = {
        "Normal": (0, 255, 0),         
        "Karies kecil": (0, 0, 255),   
        "Karies sedang": (0, 0, 130),  
        "Karies besar": (0, 0, 50), 
        "Stain": (255, 0, 255),        
        "Karang gigi": (0, 255, 255),  
        "Lain-Lain": (128, 128, 128) 
    }
    
    for i, b in enumerate(bboxes):
        obj_score, cls_index = b[4], int(b[5])
        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        label = names[cls_index]

        # Get color according to label
        color = label_colors.get(label, (255, 255, 255))  # Default to white if label not found

        # Draw bounding box with the specified color
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Determine label text position
        text_y = y1 - 5 if y1 >= 5 else y1 + 20  # Shift text down if near top edge
        if y1 < img.shape[0] // 2:  # If object is in the upper half of the image
            text_y = y2 + 20  # Place text below the object
        else:  # If object is in the lower half of the image
            text_y = y1 - 5 - 20  # Place text above the object

        cv2.putText(img, '%.2f' % obj_score, (x1, text_y), 0, 0.7, color, 2)
        cv2.putText(img, label, (x1, text_y - 20), 0, 0.7, color, 2)
        
    result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"result_{file.filename}")
    cv2.imwrite(result_image_path, img)
    
    return jsonify({'filename': os.path.basename(result_image_path)})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
