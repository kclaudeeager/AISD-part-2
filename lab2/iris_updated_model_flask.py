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
from new_model import new_model, score,update_dataset,train,models,datasets,score,test
import logging
app = Flask(__name__)
CORS(app)
executor = ThreadPoolExecutor()
logging.basicConfig(filename='backend_logs.log', level=logging.INFO)
# the minimal Flask application
@app.route('/')
def index():
    return Response( '<h1>Hello, Flask.!</h1>', status=201 )

current_datasets = datasets


@app.route('/iris/datasets', methods=['POST'])
def iris():
    logging.info('Request received for /iris/datasets')
    if 'train' in request.form:
        train = request.form['train']
        # Convert the CSV string into a DataFrame
        try:
            data = io.StringIO(train)
            train_df = pd.read_csv(data)
            logging.info('CSV data converted to DataFrame')
        except pd.errors.ParserError:
            logging.error('Invalid CSV data')
            abort(400, 'Invalid CSV data')
        # Add the DataFrame to the datasets list
        # Shuffle the dataset
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        current_datasets.append(train_df)
        update_dataset(current_datasets)
        # Return the index of the created dataset
        response = make_response(jsonify({"index": len(current_datasets) - 1}), 201)
        logging.info('Dataset uploaded successfully with index: ' + str(len(current_datasets) - 1))
        return response
    else:
        logging.error('No train data in request')
        abort(400, 'No train data in request')


UPLOAD_FOLDER = './experiments/'

@app.route('/iris/datasets-upload', methods=['POST'])
def uploadiris():
    logging.info('Request received for /iris/datasets-upload')
    if 'train' in request.files:
        train = request.files['train']
        filename = secure_filename(train.filename)
        train.save(os.path.join(UPLOAD_FOLDER, filename))
        # Add the filename to the datasets list
        # Get csv data from file
        train_df = pd.read_csv(os.path.join(UPLOAD_FOLDER, filename))
        # Shuffle the DataFrame
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        current_datasets.append(train_df)
        update_dataset(current_datasets)
        # Return the index of the created dataset
        response = make_response(jsonify({"index": len(datasets) - 1}), 201)
        logging.info('Dataset uploaded successfully with index: ' + str(len(datasets) - 1))
        return response
    else:
        logging.error('No train data in request')
        abort(400, 'No train data in request')


@app.route('/iris/model', methods=['POST'])
def create_model():
    logging.info('Request received for /iris/model')
    try:
        if 'dataset' in request.form:
            dataset_index = int(request.form['dataset'])
            
            if dataset_index < 0 or dataset_index >= len(datasets):
                logging.error('Invalid dataset index')
                abort(400, 'Invalid dataset index')
            # Offload model creation to a separate thread
            logging.info('Model creation offloaded to a separate thread')
            future = executor.submit(new_model, dataset_index)
            model_index = future.result()  # This will block until the model is created
            logging.info('Model created successfully with index: ' + str(model_index))

            print(model_index)
            response = make_response(jsonify({"model index": model_index}), 201)
            logging.info('Model created successfully with index: ' + str(model_index))
            return response
        else:
            abort(400, 'No dataset index in request')
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/iris/model/<int:model_index>', methods=['PUT'])
def retrain_model(model_index):
    print("Retrain model index: ", model_index)
    logging.info('Request received for /iris/model/' + str(model_index))
    try:
        # Get the dataset index from the query parameters
        dataset_index = request.args.get('dataset', type=int)
        print("Dataset index: ", dataset_index)
        # Check if dataset_index is None
        if dataset_index is None:
            abort(400, 'No dataset index in request')

        # Check if the dataset index is valid
        if dataset_index < 0 or dataset_index >= len(datasets):
            logging.error('Invalid dataset index')
            abort(400, 'Invalid dataset index')

        # Check if the model index is valid
        if model_index < 0 or model_index >= len(models):
            logging.error('Invalid model index')
            abort(400, 'Invalid model index')

        # Retrain the model
        # Offload model creation to a separate thread
        logging.info('Model retraining offloaded to a separate thread')
        future = executor.submit(train, model_index, dataset_index)
        history = future.result()  # This will block until the model is trained
        logging.info('Model retrained successfully')
        
        # Return the learning curve history
        response = make_response(jsonify({"history": history}), 200)
        logging.info('Model retrained successfully')
        return response
    except Exception as e:
        logging.error(str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/iris/model/<int:model_index>', methods=['GET'])
def score_model(model_index):
    logging.info('Request received for /iris/model/' + str(model_index))
    try:
        # Check if the model index is valid
        if model_index < 0 or model_index >= len(models):
            logging.error('Invalid model index')
            abort(400, 'Invalid model index')

        # Get the scoring fields from the query parameters
        fields = request.args.get('fields')
        if not fields:
            logging.error('Missing fields')
            abort(400, 'Missing fields')
   
        # Convert the fields to a list of floats
        features = list(map(float, fields.split(',')))

        # Score the model with the fields
        # Offload model scoring to a separate thread
        logging.info('Model scoring offloaded to a separate thread')
        future = executor.submit(score, model_index, features)
        score_result = future.result()  # This will block until the model is scored
        print(score_result)
        logging.info('Model scored successfully')
        # Return the score
        response = make_response(jsonify({"score_result": score_result}), 200)
    
        return response
    except Exception as e:
        logging.error(str(e))
        return jsonify({"error": str(e)}), 500

# add test route: ris/model/<n>/test?dataset=<m>
@app.route('/iris/model/<int:model_index>/test', methods=['GET'])
def test_model(model_index):
    logging.info('Request received for /iris/model/' + str(model_index) + '/test'+str(request.args))
    try:
        # Check if the model index is valid
        if model_index < 0 or model_index >= len(models):
            logging.error('Invalid model index')
            abort(400, 'Invalid model index')

        # Get the dataset index from the query parameters
        dataset_index = request.args.get('dataset', type=int)
        # Check if dataset_index is None
        if dataset_index is None:
            logging.error('No dataset index in request')
            abort(400, 'No dataset index in request')

        # Check if the dataset index is valid
        if dataset_index < 0 or dataset_index >= len(datasets):
            logging.error('Invalid dataset index')
            abort(400, 'Invalid dataset index')
        
        # Test the model
        # Offload model testing to a separate thread
        logging.info('Model testing offloaded to a separate thread')
        future = executor.submit(test, model_index, dataset_index)
        test_result = future.result()  # This will block until the model is tested
        logging.info('Model tested successfully')
        # Return the test result
        response = make_response(jsonify({"test_result": test_result}), 200)
        return response
    except Exception as e:
        logging.error(str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    logging.info('Starting Flask app')
    app.run(debug=True, host='0.0.0.0', port=4000)
