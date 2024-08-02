from flask import Flask, request, render_template, send_from_directory
import numpy as np
import os
from PIL import Image
import tensorflow as tf
import joblib
import io

app = Flask(__name__)
model = joblib.load("imgClassifierPet.pkl")

# Provided label encoding
label_encoding = {0: 'saint bernard', 1: 'bombay', 2: 'japanese chin', 3: 'great pyrenees', 
                  4: 'pomeranian', 5: 'samoyed', 6: 'keeshond', 7: 'american bulldog', 
                  8: 'basset hound', 9: 'american pit bull terrier', 10: 'scottish terrier', 
                  11: 'havanese', 12: 'english setter', 13: 'pug', 14: 'sphynx', 
                  15: 'chihuahua'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read()))
    image_resized = image.resize((224, 224))
    image_array = np.array(image_resized) / 255.0
    image_array = image_array.reshape((1, 224, 224, 3))

    prediction = model.predict(image_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class = label_encoding.get(predicted_class_index, "Unknown")

    filepath = os.path.join('static', file.filename)
    image.save(filepath)
    
    return render_template('result.html', image=file.filename, prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
