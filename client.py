import requests
import sys

SERVER_URL = 'http://localhost:4000'  # Update with your server URL
LOG_FILE = 'client_logs.txt'  # File to save console logs

def upload_dataset(train_data):
    try:
        response = requests.post(f"{SERVER_URL}/iris/datasets", data={'train': train_data})
        log_message = f"Upload dataset - Host: {SERVER_URL}, Endpoint: /iris/datasets, Status: {response.status_code}, Response: {response.text}\n"
        print(log_message)
        write_to_log(log_message)
        if response.status_code == 201:
            dataset_index = response.json()['index']
            log_message = f"Dataset uploaded successfully! Index: {dataset_index}\n"
            print(log_message)
            write_to_log(log_message)
            return dataset_index
        else:
            return None
    except Exception as e:
        log_message = f"Error uploading dataset: {str(e)}\n"
        print(log_message)
        write_to_log(log_message)
        return None

def build_model(dataset_index):
    try:
        response = requests.post(f"{SERVER_URL}/iris/model", data={'dataset': dataset_index})
        log_message = f"Build model - Host: {SERVER_URL}, Endpoint: /iris/model, Status: {response.status_code}, Response: {response.text}\n"
        print(log_message)
        write_to_log(log_message)
        if response.status_code == 201:
            model_index = response.json()['model index']
            log_message = f"Model built successfully! Index: {model_index}\n"
            print(log_message)
            write_to_log(log_message)
            return model_index
        else:
            return None
    except Exception as e:
        log_message = f"Error building model: {str(e)}\n"
        print(log_message)
        write_to_log(log_message)
        return None

def retrain_model(model_index, dataset_index):
    try:
        response = requests.put(f"{SERVER_URL}/iris/model/{model_index}?dataset={dataset_index}")
        log_message = f"Retrain model - Host: {SERVER_URL}, Endpoint: /iris/model/{model_index}, Status: {response.status_code}, Response: {response.text}\n"
        print(log_message)
        write_to_log(log_message)
        if response.status_code == 200:
            log_message = "Model retrained successfully!\n"
            print(log_message)
            write_to_log(log_message)
            return response.json()['history']
        else:
            return None
    except Exception as e:
        log_message = f"Error retraining model: {str(e)}\n"
        print(log_message)
        write_to_log(log_message)
        return None

def score_model(model_index, features):
    try:
        response = requests.get(f"{SERVER_URL}/iris/model/{model_index}?fields={','.join(map(str, features))}")
        log_message = f"Score model - Host: {SERVER_URL}, Endpoint: /iris/model/{model_index}, Status: {response.status_code}, Response: {response.text}\n"
        print(log_message)
        write_to_log(log_message)
        if response.status_code == 200:
            log_message = f"Model scored successfully! Result: {response.json()['score_result']}\n"
            print(log_message)
            write_to_log(log_message)
            return response.json()['score_result']
        else:
            return None
    except Exception as e:
        log_message = f"Error scoring model: {str(e)}\n"
        print(log_message)
        write_to_log(log_message)
        return None
# Add test function
def test_model(model_index, dataset_index):
    try:
        response = requests.get(f"{SERVER_URL}/iris/model/{model_index}/test?dataset={dataset_index}")
        log_message = f"Test model - Host: {SERVER_URL}, Endpoint: /iris/model/{model_index}/test, Status: {response.status_code}, Response: {response.text}\n"
        print(log_message)
        write_to_log(log_message)
        if response.status_code == 200:
            log_message = f"Model tested successfully!\n"
            print(log_message)
            write_to_log(log_message)
            return response.json()
        else:
            return None
    except Exception as e:
        log_message = f"Error testing model: {str(e)}\n"
        print(log_message)
        write_to_log(log_message)
        return None
    
def write_to_log(message):
    with open(LOG_FILE, 'a') as f:
        f.write(message)

def main():
    try:
        # Read dataset from file
        with open('experiments/iris_extended_encoded.csv', 'r') as file:
            train_data = file.read()
    
        
        # Upload dataset
        dataset_index = upload_dataset(train_data)
        if dataset_index is None:
            return

        # Build model
        model_index = build_model(dataset_index)
        if model_index is None:
            return

        # Retrain model (optional)
        retrain_model(model_index, dataset_index)

        # Score model
        features = features = [
            1000,  # elevation
            1,  # soil_type
            8.5,  # sepal_length
            5.3,  # sepal_width
            16,  # petal_length
            24,  # petal_width
            45.25,  # sepal_area
            384,  # petal_area
            1.6,  # sepal_aspect_ratio
            0.67,  # petal_aspect_ratio
            0.53,  # sepal_to_petal_length_ratio
            0.22,  # sepal_to_petal_width_ratio
            -7.5,  # sepal_petal_length_diff
            -18.7,  # sepal_petal_width_diff
            0.2,  # petal_curvature_mm
            100,  # petal_texture_trichomes_per_mm2
            200,  # leaf_area_cm2
            6.7,  # sepal_area_sqrt
            19.6,  # petal_area_sqrt
            0.12  # area_ratios
        ]
        score_result = score_model(model_index, features)

        if score_result is not None:
            print("Score:", score_result)
     
        # Test model
            # ask user the model id and dataset id
        model_index = int(input("Enter the model index: "))
        dataset_index = int(input("Enter the dataset index: "))
        test_result = test_model(model_index, dataset_index)
        if test_result is not None:
            print("Test result:", test_result)
    except Exception as e:
        log_message = "Error in main function: {}\n".format(e)
        print(log_message)
        write_to_log(log_message)

if __name__ == "__main__":
    main()
