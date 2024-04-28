
from decimal import Decimal
import json
from turtle import pd
from datetime import datetime
from lab4.lab4_header import scores_table
from tqdm import tqdm
import random
import uuid

def post_score(log_table, feature_string, class_string, actual_string, prob_string):
    current_time = datetime.utcnow().strftime('%H:%M:%S.%f')[:-3]
    unique_id = str(uuid.uuid4())  # Generate a unique ID
    # print("partition_key: ", current_time)
    # print("sort_key: ", unique_id)
    response = log_table.put_item(
        Item={
            'partition_key': current_time,
            'sort_key': unique_id,  # Use the unique ID as the sort key
            'Features': feature_string,
            'Class' : class_string,
            'Actual' : actual_string,
            'Probability' : str(prob_string)
        }
    )
    
    return response
    # except Exception as e:
    #     print(f" Error in post_score: {str(e)}")
    #     return str(e)

def back_end_test(data,response,probabilities):
    '''  
    A funtion which prepares inputs as needed by the post_score() function and to call that function for each record in the test dataframe

    '''
    #try:
    print("Inside back_end_test")
    test_result = response.json()["test_result"]
            
    # Extract accuracy, loss, model ID, and dataset ID
    #accuracy = test_result['accuracy']
    # loss = test_result['loss']
    # model_id = test_result['model_id']
    # dataset_id = test_result['dataset_id']

    # Extract actual and predicted classes
    actual_classes = test_result['actual_classes']
    predicted_classes = test_result['predicted_classes']
    data = json.loads(data)

 

    for i in tqdm(range(len(test_result['actual_classes']))):
        feature_string = str(data[i])
        class_string = predicted_classes[i]
        actual_string = actual_classes[i]
        classes_prob = probabilities[i]

        # Randomly modify class_string and classes_prob
        if random.random() < 0.1:  # 10% chance of modification
            class_string = str(random.randint(1, 3))  # Randomly choose between 1, 2, and 3
        elif random.random() < 0.3:
             classes_prob = [random.uniform(0, 0.9) for _ in classes_prob]  # Randomly choose a probability below 0.9

        highest_prob = str(max(classes_prob))

        # Call the post_score function
        post_score(scores_table, feature_string, class_string, actual_string, highest_prob)

        print("Succeeded in posting the score for record ", i+1)
    print("Succeeded in posting all the scores")
    return "Success"
    # except Exception as e:
    #     print(f" Error: {str(e)}")
    #     return str(e)


        


   

