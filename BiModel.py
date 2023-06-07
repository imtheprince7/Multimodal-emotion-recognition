from flask import Flask,render_template,request
import sys
import os

sys.path.append(os.path.abspath('BiModel/'))
import FeatureExtractionForBiModel

app=Flask(__name__)
@app.route('/')
def index():
    return render_template('biModel.html')
@app.route('/checkLabel',methods=['POST'])
def check():
    if request.method=='POST':

        text=request.form['text']
        audio=request.files['audio']

        audio.save(audio.filename)
        result = FeatureExtractionForBiModel.Feature(text,audio.filename)[0]
        os.remove(audio.filename)

    return render_template('biModel.html',res=result)
app.run(debug=True)

