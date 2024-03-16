from flask import Flask
from flask import request
from flask import Response
from flask import abort
from flask import jsonify
from flask import make_response
from sklearn import datasets
from werkzeug.utils import secure_filename
import os
import io
import pandas as pd
import numpy as np
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor
from new_model import new_model, score,update_dataset,train,models,datasets,score
app = Flask(__name__)
CORS(app)
executor = ThreadPoolExecutor()

# the minimal Flask application
@app.route('/')
def index():
    return Response( '<h1>Hello, Flask.!</h1>', status=201 )

current_datasets = datasets


@app.route('/iris/datasets', methods=['POST'])
def iris():
    if 'train' in request.form:
        train = request.form['train']
        # Convert the CSV string into a DataFrame
        try:
            data = io.StringIO(train)
            train_df = pd.read_csv(data)
        except pd.errors.ParserError:
            abort(400, 'Invalid CSV data')
        # Add the DataFrame to the datasets list
        current_datasets.append(train_df)
        update_dataset(current_datasets)
        # Return the index of the created dataset
        response = make_response(jsonify({"index": len(current_datasets) - 1}), 201)
        return response
    else:
        abort(400, 'No train data in request')


UPLOAD_FOLDER = './experiments/'

@app.route('/iris/datasets-upload', methods=['POST'])
def uploadiris():
    if 'train' in request.files:
        train = request.files['train']
        filename = secure_filename(train.filename)
        train.save(os.path.join(UPLOAD_FOLDER, filename))
        # Add the filename to the datasets list
        current_datasets.append(filename)
        # Return the index of the created dataset
        response = make_response(jsonify({"index": len(datasets) - 1}), 201)
        return response
    else:
        abort(400, 'No train data in request')


@app.route('/iris/model', methods=['POST'])
def create_model():
    try:
        if 'dataset' in request.form:
            dataset_index = int(request.form['dataset'])
            
            if dataset_index < 0 or dataset_index >= len(datasets):
                abort(400, 'Invalid dataset index')
            # Offload model creation to a separate thread
            future = executor.submit(new_model, dataset_index)
            model_index = future.result()  # This will block until the model is created

            print(model_index)
            response = make_response(jsonify({"model index": model_index}), 201)
            return response
        else:
            abort(400, 'No dataset index in request')
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/iris/model/<int:model_index>', methods=['PUT'])
def retrain_model(model_index):
    print("Retrain model index: ", model_index)
    try:
        # Get the dataset index from the query parameters
        dataset_index = request.args.get('dataset', type=int)
        print("Dataset index: ", dataset_index)
        # Check if dataset_index is None
        if dataset_index is None:
            abort(400, 'No dataset index in request')

        # Check if the dataset index is valid
        if dataset_index < 0 or dataset_index >= len(datasets):
            abort(400, 'Invalid dataset index')

        # Check if the model index is valid
        if model_index < 0 or model_index >= len(models):
            abort(400, 'Invalid model index')

        # Retrain the model
        # Offload model creation to a separate thread
        future = executor.submit(train, model_index, dataset_index)
        history = future.result()  # This will block until the model is trained
        
        # Return the learning curve history
        response = make_response(jsonify({"history": history}), 200)
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/iris/model/<int:model_index>', methods=['GET'])
def score_model(model_index):
    try:
        # Check if the model index is valid
        if model_index < 0 or model_index >= len(models):
            abort(400, 'Invalid model index')

        # Get the scoring fields from the query parameters
        fields = request.args.get('fields')
        if not fields:
            abort(400, 'Missing fields')
   
        # Convert the fields to a list of floats
        features = list(map(float, fields.split(',')))

        # Score the model with the fields
        future = executor.submit(score, model_index, features)
        score_result = future.result()  # This will block until the model is scored
        print(score_result)
        # Return the score
        response = make_response(jsonify({"score_result": score_result}), 200)
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4000)
