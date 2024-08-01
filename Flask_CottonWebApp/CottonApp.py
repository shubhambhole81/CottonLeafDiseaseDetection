from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
import cv2
import re
import mysql.connector
import joblib
from image_function import *


app = Flask(__name__)

fixed_size = tuple((500, 500))

# Loading model
Disease_type_model = joblib.load('Disease_type_detection.sav')

# Reading CSV file for disease cure
data = pd.read_excel('disease_control.xlsx')
disease_control = data.set_index('Disease')['Control'].to_dict()

host = 'localhost'
user = 'root'
password = 'root'
database = 'profile'
con = mysql.connector.connect(host=host,user=user, password=password, database=database)
cursor = con.cursor()


@app.route('/login', methods =['GET', 'POST'])
def login():
	msg = ''
	if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
		username = request.form['username']
		password = request.form['password']
		cursor.execute('SELECT * FROM user WHERE username =%s  AND password=%s', (username, password, ))
		account = cursor.fetchone()
		if account:
			msg = 'Logged in successfully !'
			return render_template('index.html', msg = msg)
		else:
			msg = 'Incorrect username / password !'
	return render_template('login.html', msg = msg)


@app.route('/register', methods =['GET', 'POST'])
def register():
	msg = ''
	if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form :
		username = request.form['username']
		password = request.form['password']
		email = request.form['email']
		cursor = con.cursor()
		cursor.execute('SELECT * FROM user WHERE username =%s', (username, ))
		account = cursor.fetchone()
		if account:
			msg = 'Account already exists !'
		elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
			msg = 'Invalid email address !'
		elif not re.match(r'[A-Za-z0-9]+', username):
			msg = 'Username must contain only characters and numbers !'
		elif not username or not password or not email:
			msg = 'Please fill out the form !'
		else:
			cursor.execute('INSERT INTO user VALUES (NULL,%s,%s,%s)', (username, password, email, ))
			con.commit()
			msg = 'You have successfully registered ! Now, Go to the login page'
	elif request.method == 'POST':
		msg = 'Please fill out the form !'
	return render_template('register.html', msg = msg)



@app.route('/detect', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        img = request.files['images']
        img.save("img.jpg")
        
        image = cv2.imread('img.jpg')

        image = cv2.resize(image, fixed_size)

        # Call for Image functions
        
        RGB_BGR       = rgb_bgr(image)
        BGR_HSV       = bgr_hsv(RGB_BGR)
        IMG_SEGMENT   = img_segmentation(RGB_BGR,BGR_HSV)

        # Call for Global Fetaures
        
        fv_hu_moments = fd_hu_moments(IMG_SEGMENT)
        fv_haralick   = fd_haralick(IMG_SEGMENT)
        fv_histogram  = fd_histogram(IMG_SEGMENT)
        
        # Concatenate 
        
        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
        
        features = []
        features.append(global_feature)

        pred_type = Disease_type_model.predict(features)
        
        # Giving Predictions
        for pred in pred_type:
            if pred == 0:
                return render_template('predict.html',detection = 'Infected', type_detection = 'Bacterial Blight', cure = disease_control['Bacterial_blight'])
            elif pred == 1:
                return render_template('predict.html',detection = 'Infected', type_detection = 'Curl Virus', cure = disease_control['Curl_virus'])
            elif pred == 2:
                return render_template('predict.html',detection = 'Infected', type_detection = 'Fussarium Wilt', cure = disease_control['Fussarium_wilt'])
            else: 
                return render_template('detect.html', detection = 'Healthy')
    return None

@app.route('/logout')
def logout():
	return redirect(url_for('login'))

@app.route('/')
def start():
    return render_template('start.html')


if __name__ == '__main__':
    app.run(debug=True)
