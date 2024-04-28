from datetime import datetime
import os
import boto3
from botocore.config import Config


# Load the environment variables
from dotenv import load_dotenv
load_dotenv()

user_name = os.getenv('USER_NAME')
print(f'Hello {user_name}!')

password = os.getenv('PASSWORD')

Access_key_ID = os.getenv('Access_key_ID')

Secret_access = os.getenv('Secret_access')

region = os.getenv('Region')

my_config = Config(
    region_name = region,
)

# Get the service resource.

session = boto3.Session(
    aws_access_key_id= Access_key_ID,
    aws_secret_access_key= Secret_access,
)

dynamodb = session.resource('dynamodb', config=my_config)

scores_table = dynamodb.Table('IrisExtended')
# ddb = session.client('dynamodb', config=my_config,)
# print(ddb.describe_limits())