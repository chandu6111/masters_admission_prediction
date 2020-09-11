import numpy as np 
from flask import Flask,request,jsonify,render_template
import pickle

app=Flask(__name__)
model=pickle.load(open('regressor.pkl','rb'))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    features=[str(x) for x in request.form.values()]
    gre=int(features[0])
    toefl=int(features[1])
    urating=int(features[2])
    sop=float(features[3])
    lor=float(features[4])
    cgpa=float(features[5])
    research=int(features[6])
    final_features=[]

    final_features.append(gre)
    final_features.append(toefl)
    final_features.append(urating)
    final_features.append(sop)
    final_features.append(lor)
    final_features.append(cgpa)
    final_features.append(research)
    print(len(final_features))
    prediction=model.predict([final_features])
    output=round(prediction[0],2)
    return render_template('index.html',prediction_text=output)


if __name__ == "__main__":
    app.run(debug=True)