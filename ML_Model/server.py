"""
This code takes the JSON data while POST request an performs the prediction using loaded model and returns
the results in JSON format.
"""

# Import libraries
import numpy as np
import flask
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)


@app.route('/')
def index():
    return flask.render_template('index.html')


# Load the model
model = pickle.load(open('TrainPickle.pkl', 'rb'))


# prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 2)
    loaded_model = pickle.load(open("TrainPickle.pkl", "rb"))
    results = loaded_model.predict(to_predict)
    return results[0]


@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        predict_list = request.form.to_dict()
        predict_list = list(predict_list.values())
        predict_list = list(map(int, predict_list))
        results = ValuePredictor(predict_list)

        if int(results) == 1:
            prediction = 'User will click the Ad'
        else:
            prediction = 'User will not click the Ad'

        return render_template("result.html", prediction=prediction)


if __name__ == '__main__':
    app.run(port=5003, debug=True)
