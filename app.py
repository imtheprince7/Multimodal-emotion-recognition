from flask import Flask,render_template,request
import sys  
import os
import numpy as np
sys.path.append(os.path.abspath('Uni_Model/'))
sys.path.append(os.path.abspath('Z__FUSION/'))
from sklearn.preprocessing import MaxAbsScaler
import AudioFeatures
from sklearn.preprocessing import LabelEncoder
import LSTM
import pandas as pd
from keras.models import load_model
model=load_model('AudioModel')
app=Flask(__name__)
data = pd.read_csv('Z__FUSION/features.csv')

# Separate features and labels
features = data.iloc[:, :-1].values
labels = data.iloc[:, -1].values
LE=LabelEncoder()
labels=LE.fit_transform(labels)
@app.route('/')
def index():
    return render_template('index.html',title='Index')
@app.route('/audiotemp')
def index1():
    return render_template('audio.html',title='Audio')
@app.route('/text',methods=['POST'])
def text():
    if request.method=='POST':
        text=request.form['text']
        res=LSTM.Test(text)
        print(res)
    return render_template('index.html',title='Index',res=res[0])
@app.route('/audio',methods=['POST'])
def audio():
    if request.method=='POST':
        file=request.files['file']
        file.save(file.filename)
        feature=AudioFeatures.extract_features(file.filename)
        print(feature)
        
        feature=np.array(feature).reshape(1,-1)
        feature=MaxAbsScaler().fit_transform(feature)
        res=np.argmax(model.predict(feature))
        res=LE.inverse_transform([res])
        print(model.predict(feature))
        
        print(res)
    return render_template('audio.html',title='Audio',res=res)

app.run(debug=True)
