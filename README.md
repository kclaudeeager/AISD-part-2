# AISD-part-2-Lab1

This project is a Flask API for building, retraining, and scoring an Iris model.

## Installation

1. Clone this repository.
2. Install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask server:

```bash
python iris_updated_model_flask.py
```

2. Use the following endpoints to interact with the API:

- POST `/iris/datasets`: Upload a new dataset.
- POST `/iris/datasets-upload`: Upload a new dataset from a file.
- POST `/iris/model`: Build a new model.
- PUT `/iris/model/<int:model_index>`: Retrain a model with a dataset.
- GET `/iris/model/<int:model_index>`: Score a model with a set of features.

## Testing

You can test the API using the provided Python client:

```bash
python client.py
```

This client uses the `requests` library to make requests to the API and logs the responses to the console.

## Logging

The Flask application logs informational and error messages to a file named `backend_logs.log`. You can view these logs to monitor the application's activity and troubleshoot any issues. To view the client logs, look into the file named `client_logs.txt`.

## API Documentation

If you want to see the API documentation, you can access the postman collection [here](https://documenter.getpostman.com/view/15574800/2sA2xpU9pj#fe818edc-69b9-496e-801e-84402488fc3b).

