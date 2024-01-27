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

def lambda_handler(event, context):
    print('event: ', event)

    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        # translate utf8
        key = unquote_plus(record['s3']['object']['key']) # url decoding
        print('bucket: ', bucket)
        print('key: ', key)
        
        eventName = record['eventName']        
        if eventName == 'ObjectRemoved:Delete':
            file_type = key[key.rfind('.')+1:len(key)]
            category = file_type
            documentId = category + "-" + key
            documentId = documentId.replace(' ', '_') # remove spaces
            documentId = documentId.replace(',', '_') # remove commas # not allowed: [ " * \\ < | , > / ? ]
            documentId = documentId.replace('/', '_') # remove slash
            documentId = documentId.lower() # change to lowercase
            # print('documentId: ', documentId)
                    
            # delete document index of opensearch
            index_name = "idx-"+documentId
            delete_index_if_exist(index_name)        
    return {
        'statusCode': 200
    }
