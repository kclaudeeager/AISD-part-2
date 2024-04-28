import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import datasets, models

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from tensorflow.keras.optimizers import RMSprop
import io
import os
import joblib
print('starting up iris model service')

global models, datasets,metrics

models = []
datasets = []
metrics = []

if os.path.exists('datasets.pkl'):
    with open('datasets.pkl', 'rb') as f:
        datasets = joblib.load(f)
        print("Datasets loaded from file")
if os.path.exists('metrics.pkl'):
    with open('metrics.pkl', 'rb') as f:
        metrics = joblib.load(f)

model_ID = 0

while os.path.exists(f'model_{model_ID}'):
    models.append(tf.keras.models.load_model(f'model_{model_ID}'))
    model_ID += 1

def update_dataset(input_datasets=None):
    global datasets
    if input_datasets is not None:
        datasets = input_datasets

    # Save the datasets to a file
    with open('datasets.pkl', 'wb') as f:
        joblib.dump(datasets, f)

def update_metrics(input_metrics=None):
    global metrics
    if input_metrics is not None:
        metrics = input_metrics

    # Save the metrics to a file
    with open('metrics.pkl', 'wb') as f:
        joblib.dump(metrics, f)

def build():
    global models

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(20,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    optimizer = RMSprop(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    models.append( model )
    model_ID = len(models) - 1

    # Save the model to a file
    model.save(f'model_{model_ID}', save_format='tf')

    return model_ID

def load_local():
    global datasets

    print("load local default data")
    if len(datasets) > 0:
        return len( datasets ) - 1
    
    dataFolder = './experiments/'
    dataFile = dataFolder + "iris_extended_encoded.csv"

    datasets.append( pd.read_csv(dataFile) )
    return len( datasets ) - 1

def add_dataset( df ):
    global datasets

    datasets.append( df )
    return len( datasets ) - 1

def get_dataset( dataset_ID ):
    global datasets

    return datasets[dataset_ID]

def train(model_ID, dataset_ID):
    global datasets, models
    print("Dataset length: ", len(datasets))
    dataset = datasets[dataset_ID]
    model = models[model_ID]
    
    X = dataset.iloc[:,1:].values
    y = dataset.iloc[:,0].values
    
    encoder =  LabelEncoder()
    y1 = encoder.fit_transform(y)
    Y = pd.get_dummies(y1).values
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

    history = model.fit(X_train, y_train, batch_size=1, epochs=10)
    print(history.history)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

    y_pred = model.predict(X_test)

    actual = np.argmax(y_test,axis=1)
    predicted = np.argmax(y_pred,axis=1)
    print(f"Actual: {actual}")
    print(f"Predicted: {predicted}")

    conf_matrix = confusion_matrix(actual, predicted)
    print('Confusion matrix on test data is {}'.format(conf_matrix))
    print('Precision Score on test data is {}'.format(precision_score(actual, predicted, average=None)))
    print('Recall Score on test data is {}'.format(recall_score(actual, predicted, average=None)))
    # Save the trained model
    model.save(f'model_{model_ID}', save_format='tf')
    return(history.history)

def new_model(dataset_ID):
    model_ID = build()
    history = train(model_ID, dataset_ID)
    print(history)
    return model_ID

def score(model_ID, features):
    global models
    model = models[model_ID]

    x_test2 = [features]  # Adjusted to handle a list of input features
    print("Type of x_test2: ", type(x_test2))
    # print the dimensions of the input
    print("Shape of x_test2: ", np.array(x_test2).shape)
    y_pred2 = model.predict(x_test2)
    print(y_pred2)
    iris_class = np.argmax(y_pred2, axis=1)[0]
    print(iris_class)

    return "Score done, class=" + str(iris_class)

def test(model_id, dataset_id):
    global models, datasets, metrics
    print("Dataset length: ", len(datasets))
    print("Model length: ", len(models))
    print("Model ID: ", model_id)
    print("Dataset ID: ", dataset_id)
    try:
        dataset = datasets[dataset_id]
        print("Dataset shape: ", dataset.shape)
    except Exception as e:
        print("Error loading dataset: ", str(e))
        return None
    
    try:
        model = models[model_id]
        print("Model loaded")
    except Exception as e:
        print("Error loading model: ", str(e))
        return None
    
    # Created a json object to convert dataframe into json of the test data

    test_data = dataset.to_json(orient='records')
    print("Test data: ", test_data)

    X = dataset.iloc[:,1:].values
    y = dataset.iloc[:,0].values
    

    encoder =  LabelEncoder()
    y1 = encoder.fit_transform(y)
    Y = pd.get_dummies(y1).values
    print("Y shape: ", Y.shape)
    print(Y)

    loss, accuracy = model.evaluate(X, Y, verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)
    y_pred = model.predict(X)

    probabilities = y_pred.tolist()
    # print("Probabilities: ", probabilities)
 
    actual = np.argmax(Y, axis=1)
    predicted = np.argmax(y_pred, axis=1)
    print(f"Actual: {actual}")
    print(f"Predicted: {predicted}")
    conf_matrix = confusion_matrix(actual, predicted)
    print('Confusion matrix on test data is {}'.format(conf_matrix))
    precision = precision_score(actual, predicted, average=None)
    recall = recall_score(actual, predicted, average=None)
    print('Precision Score on test data is {}'.format(precision))
    print('Recall Score on test data is {}'.format(recall))
    
    # Save the metrics in a dictionary and append it to the metrics list
    test_results = {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "loss": loss,
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "actual_classes": actual.tolist(),  # Add actual classes to the dictionary
        "predicted_classes": predicted.tolist()  # Add predicted classes to the dictionary
    }

    print("Test results gotten: ", test_results)
    for i, metric in enumerate(metrics):
        if metric["model_id"] == model_id and metric["dataset_id"] == dataset_id:
            metrics[i] = test_results  # Update the test results
            break
    else:
        metrics.append(test_results)  # Append the test results
    
    return test_results,test_data,probabilities  # Return the test results and test data