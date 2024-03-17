import os
from flask import Flask, request, render_template, jsonify
from keras.models import load_model
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model

app = Flask(__name__)
model = load_model("models/model_19.h5")

resnet50_model = ResNet50(weights='imagenet', input_shape=(224, 224, 3))
resnet50_model = Model(resnet50_model.input, resnet50_model.layers[-2].output)

word_to_index = pickle.load(open("word_to_idx.pkl", "rb"))
index_to_word = pickle.load(open("idx_to_word.pkl", "rb"))

def preprocess_image(img):
    img = image.load_img(img, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def encode_image(img):
    img = preprocess_image(img)
    feature_vector = resnet50_model.predict(img)
    return feature_vector

def predict_caption(photo):
    inp_text = "startseq"
    for i in range(38):
        sequence = [word_to_index[w] for w in inp_text.split() if w in word_to_index]
        sequence = pad_sequences([sequence], maxlen=38, padding='post')
        ypred = model.predict([photo, sequence])
        ypred = ypred.argmax()
        word = index_to_word[ypred]
        inp_text += (' ' + word)
        if word == 'endseq':
            break
    final_caption = inp_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        filename = os.path.join("static/uploads", file.filename)
        file.save(filename)
        photo = encode_image(filename).reshape((1, 2048))
        caption = predict_caption(photo)
        os.remove(filename)

        return jsonify({'caption': caption, 'image_url': f"/{filename}"})  

if __name__ == '__main__':
    os.makedirs("static/uploads", exist_ok=True) 
    app.run(debug=True)
