import numpy as np 
from flask import Flask,request,jsonify,render_template
import pickle
#import os
import warnings

app=Flask(__name__)
warnings.filterwarnings("ignore")
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict():
    if request.method=='POST':
        features=[str(x) for x in request.form.values()]
        gre=int(features[0])
        toefl=int(features[1])
        ur=float(features[2])
        sop=float(features[3])
        lor=float(features[4])
        cgpa=float(features[5])
        research=int(features[6])
        final_features=[]
        final_features.append(gre)
        final_features.append(toefl)
        final_features.append(ur)
        final_features.append(sop)
        final_features.append(lor)
        final_features.append(cgpa)
        final_features.append(research)
        lr_model=pickle.load(open("model.pkl","rb"))

        prediction=lr_model.predict([final_features])
        output=round(prediction[0],2)
        output=output*100;
    return render_template('index.html',prediction_text=output)


if __name__ == "__main__":
   # from waitress import serve
    app.run(debug=False)