from flask import Flask, request,  jsonify
from flask import make_response
from FBModel import FBModel
import pandas as pd
import codecs, json 
import numpy as np

app = Flask(__name__)

from flask import request

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.route('/predict', methods=['GET'])
def fb_predict():
    url = 'posts-predict.json'
    df_predict = pd.read_json(url, orient='columns')
    model_class = FBModel()
    model_class.preprocess(df_predict)
    y_pred = model_class.predict(df_predict,'FB-Model-01')
    print(df_predict['shares'].values)
    print(y_pred)

#    return request.json['test']
    # task = {
    #     'score': '100'
    # }
#    print(json.dumps(y_pred, cls=NumpyEncoder))
    return jsonify({'prediction': json.dumps(y_pred, cls=NumpyEncoder)}), 201

if __name__ == '__main__':
    app.run(debug=True)

