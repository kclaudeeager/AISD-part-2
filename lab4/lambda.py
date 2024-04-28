import json
import logging
import boto3

# Get the service resource.

client = boto3.client('dynamodb')
def lambda_handler(event, context):
    print(event)
    for rec in event['Records']:
        print(rec)
        if rec['eventName'] == 'INSERT':
            UpdateItem = rec['dynamodb']['NewImage']
            print(UpdateItem)

            # Check if the predicted class does not match the actual class
            if UpdateItem['Class']['N'] != UpdateItem['Actual']['N']:
                response = client.put_item(TableName='IrisExtendedRetrain', Item=UpdateItem)
                print(response)

            # Check if the prediction confidence is below 0.9
            elif float(UpdateItem['Probability']['S']) < 0.9:
                response = client.put_item(TableName='IrisExtendedRetrain', Item=UpdateItem)
                print(response)

    return {
        'statusCode': 200,
        'body': json.dumps('IrisExtendedRetrain Lambda return')
    }