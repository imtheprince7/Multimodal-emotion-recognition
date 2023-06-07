from flask import Flask,render_template,request
import sys
import os
sys.path.append(os.path.abspath('MultiModal/'))
import FeatureExtractionForModel

app=Flask(__name__)
@app.route('/')
def index():
    return render_template('multimodel.html')
@app.route('/checkLabel',methods=['POST'])
def check():
    if request.method=='POST':

        text=request.form['text']
        audio=request.files['audio']
        video=request.files['video']

        audio.save(audio.filename)
        video.save(video.filename)
        result = FeatureExtractionForModel.Feature(text,video.filename,audio.filename)[0]
        os.remove(audio.filename)
        os.remove(video.filename)

    return render_template('multimodel.html',res=result)
app.run(debug=True)

