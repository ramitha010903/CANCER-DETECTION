from flask import Flask, render_template, request, redirect, url_for, session, flash
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
import json

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

# Load the Keras model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Function to make prediction
def predict_image(image_path):
    size = (224, 224)
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    return class_name, confidence_score

# Utility to load user data from JSON
def load_users():
    with open('users.json', 'r') as f:
        return json.load(f)

# Utility to save user data to JSON
def save_user(username, password):
    users = load_users()
    users[username] = password
    with open('users.json', 'w') as f:
        json.dump(users, f)

# Route for landing page
@app.route('/')
def landing():
    return render_template('landing.html')

# Route for login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = load_users()

        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password')
            return redirect(url_for('login'))
    return render_template('login.html')

# Route for registration page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = load_users()

        if username in users:
            flash('Username already exists')
            return redirect(url_for('register'))
        else:
            save_user(username, password)
            session['username'] = username
            return redirect(url_for('index'))
    return render_template('register.html')

# Route for home page (image upload)
@app.route('/home')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        image_path = os.path.join('static', file.filename)
        file.save(image_path)

        class_name, confidence_score = predict_image(image_path)

        return render_template('result.html', class_name=class_name, confidence=confidence_score, image_path=file.filename)

if __name__ == '__main__':
    app.run(debug=True)
