import json
import boto3
import os
import traceback
from botocore.config import Config
from urllib.parse import unquote_plus

s3 = boto3.client('s3')
s3_bucket = os.environ.get('s3_bucket') # bucket name
meta_prefix = "metadata/"

opensearch_account = os.environ.get('opensearch_account')
opensearch_passwd = os.environ.get('opensearch_passwd')
opensearch_url = os.environ.get('opensearch_url')

from opensearchpy import OpenSearch
def delete_index_if_exist(index_name):
    client = OpenSearch(
        hosts = [{
            'host': opensearch_url.replace("https://", ""), 
            'port': 443
        }],
        http_compress = True,
        http_auth=(opensearch_account, opensearch_passwd),
        use_ssl = True,
        verify_certs = True,
        ssl_assert_hostname = False,
        ssl_show_warn = False,
    )

    if client.indices.exists(index_name):
        print('delete opensearch document index: ', index_name)
        response = client.indices.delete(
            index=index_name
        )
        print('removed index: ', response)    
    else:
        print('no index: ', index_name)

# load csv documents from s3
def lambda_handler(event, context):
    print('event: ', event)

    documentIds = []
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        # translate utf8
        key = unquote_plus(record['s3']['object']['key']) # url decoding
        print('bucket: ', bucket)
        print('key: ', key)

        # get metadata from s3
        metadata_key = meta_prefix+key+'.metadata.json'
        print('metadata_key: ', metadata_key)

        metadata_obj = s3.get_object(Bucket=bucket, Key=metadata_key)
        metadata_body = metadata_obj['Body'].read().decode('utf-8')
        metadata = json.loads(metadata_body)
        print('metadata: ', metadata)
        documentId = metadata['DocumentId']
        print('documentId: ', documentId)
        documentIds.append(documentId)

        # delete document index of opensearch
        index_name = "rag-index-"+documentId
        # print('index_name: ', index_name)
        delete_index_if_exist(index_name)
    
    return {
        'statusCode': 200
    }
