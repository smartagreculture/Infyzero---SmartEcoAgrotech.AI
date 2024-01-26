from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__) 

with open(r"crop_recommendation_rf_model.pkl", "rb") as input_file:
    model = pickle.load(input_file)

with open(r"crop_recommendation_label_encoder.pkl", "rb") as input_file:
    label_encoder = pickle.load(input_file)

@app.route("/")
def hello():
  return "Hello World!"

@app.route('/crop-recommendation', methods = ['POST'])
def crop_recommendation():
    try:
        data = request.get_json()
        df = pd.DataFrame(data)
        predicted_label = label_encoder.inverse_transform(model.predict(df))
        predictions_proba = model.predict_proba(df)
        max_prob_indexes = np.argmax(predictions_proba, axis = 1)
        max_prob = predictions_proba[np.arange(len(predictions_proba)), max_prob_indexes]
        print(predicted_label, max_prob)
        result = {'predicted_label': predicted_label[0], 'prediction_proba': max_prob[0]}
        return jsonify(result)
    except Exception as e:
        return jsonify({'Error': str(e)}), 500

app.run()

# {
#     "N": 31,
#     "P": 53,
#     "K": 16,
#     "temperature": 28.742,
#     "humidity": 85.816,
#     "ph": 6.452,
#     "rainfall": 48.545
# }
