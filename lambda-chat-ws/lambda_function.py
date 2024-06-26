import json
import boto3
import os
import time
import datetime
from io import BytesIO
import traceback
import re
from urllib import parse

from botocore.config import Config
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain_community.docstore.document import Document
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.opensearch_vector_search import OpenSearchVectorSearch
from langchain_community.embeddings import BedrockEmbeddings
from multiprocessing import Process, Pipe
from opensearchpy import OpenSearch

from langchain_community.chat_models import BedrockChat
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_aws import ChatBedrock

s3 = boto3.client('s3')
s3_bucket = os.environ.get('s3_bucket') # bucket name
s3_prefix = os.environ.get('s3_prefix')
meta_prefix = "metadata/"
callLogTableName = os.environ.get('callLogTableName')
profile_of_LLMs = json.loads(os.environ.get('profile_of_LLMs'))
isReady = False   

opensearch_account = os.environ.get('opensearch_account')
opensearch_passwd = os.environ.get('opensearch_passwd')
enableReference = os.environ.get('enableReference', 'false')
debugMessageMode = os.environ.get('debugMessageMode', 'false')
opensearch_url = os.environ.get('opensearch_url')
path = os.environ.get('path')
doc_prefix = s3_prefix+'/'

roleArn = os.environ.get('roleArn')
top_k = int(os.environ.get('numberOfRelevantDocs', '8'))
selected_LLM = 0
MSG_LENGTH = 100
MSG_HISTORY_LENGTH = 20
speech_generation = True
history_length = 0
token_counter_history = 0

enableNoriPlugin = os.environ.get('enableNoriPlugin')
enableParallelSummay = os.environ.get('enableParallelSummay')

os_client = OpenSearch(
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

# websocket
connection_url = os.environ.get('connection_url')
client = boto3.client('apigatewaymanagementapi', endpoint_url=connection_url)
print('connection_url: ', connection_url)

HUMAN_PROMPT = "\n\nHuman:"
AI_PROMPT = "\n\nAssistant:"

map_chain = dict() 

# Multi-LLM
def get_chat(profile_of_LLMs, selected_LLM):
    profile = profile_of_LLMs[selected_LLM]
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    print(f'LLM: {selected_LLM}, bedrock_region: {bedrock_region}, modelId: {modelId}')
    maxOutputTokens = int(profile['maxOutputTokens'])
                          
    # bedrock   
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=bedrock_region,
        config=Config(
            retries = {
                'max_attempts': 30
            }            
        )
    )
    parameters = {
        "max_tokens":maxOutputTokens,     
        "temperature":0.1,
        "top_k":250,
        "top_p":0.9,
        "stop_sequences": [HUMAN_PROMPT]
    }
    # print('parameters: ', parameters)

    chat = ChatBedrock(  
        model_id=modelId,
        client=boto3_bedrock, 
        model_kwargs=parameters,
    )       
    return chat

def get_embedding(profile_of_LLMs, selected_LLM):
    profile = profile_of_LLMs[selected_LLM]
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    print(f'Embedding: {selected_LLM}, bedrock_region: {bedrock_region}, modelId: {modelId}')
    
    # bedrock   
    boto3_bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=bedrock_region,
        config=Config(
            retries = {
                'max_attempts': 30
            }            
        )
    )
    
    bedrock_embedding = BedrockEmbeddings(
        client=boto3_bedrock,
        region_name = bedrock_region,
        model_id = 'amazon.titan-embed-text-v1' 
    )  
    
    return bedrock_embedding

def sendMessage(id, body):
    try:
        client.post_to_connection(
            ConnectionId=id, 
            Data=json.dumps(body)
        )
    except Exception:
        err_msg = traceback.format_exc()
        print('err_msg: ', err_msg)
        raise Exception ("Not able to send a message")

def sendResultMessage(connectionId, requestId, msg):    
    result = {
        'request_id': requestId,
        'msg': msg,
        'status': 'completed'
    }
    #print('debug: ', json.dumps(debugMsg))
    sendMessage(connectionId, result)

def sendDebugMessage(connectionId, requestId, msg):
    debugMsg = {
        'request_id': requestId,
        'msg': msg,
        'status': 'debug'
    }
    #print('debug: ', json.dumps(debugMsg))
    sendMessage(connectionId, debugMsg)

def sendErrorMessage(connectionId, requestId, msg):
    errorMsg = {
        'request_id': requestId,
        'msg': msg,
        'status': 'error'
    }
    print('error: ', json.dumps(errorMsg))
    sendMessage(connectionId, errorMsg)

def isKorean(text):
    # check korean
    pattern_hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')
    word_kor = pattern_hangul.search(str(text))
    # print('word_kor: ', word_kor)

    if word_kor and word_kor != 'None':
        print('Korean: ', word_kor)
        return True
    else:
        print('Not Korean: ', word_kor)
        return False

def general_conversation(connectionId, requestId, chat, query):
    global time_for_inference, history_length, token_counter_history    
    time_for_inference = history_length = token_counter_history = 0
    
    if debugMessageMode == 'true':  
        start_time_for_inference = time.time()
    
    if isKorean(query)==True :
        system = (
            "다음의 Human과 Assistant의 친근한 이전 대화입니다. Assistant은 상황에 맞는 구체적인 세부 정보를 충분히 제공합니다. Assistant의 이름은 서연이고, 모르는 질문을 받으면 솔직히 모른다고 말합니다."
        )
    else: 
        system = (
            "Using the following conversation, answer friendly for the newest question. If you don't know the answer, just say that you don't know, don't try to make up an answer. You will be acting as a thoughtful advisor."
        )
    
    human = "{input}"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), MessagesPlaceholder(variable_name="history"), ("human", human)])
    print('prompt: ', prompt)
    
    history = memory_chain.load_memory_variables({})["chat_history"]
    print('memory_chain: ', history)
                
    chain = prompt | chat    
    try: 
        isTyping(connectionId, requestId)  
        stream = chain.invoke(
            {
                "history": history,
                "input": query,
            }
        )
        msg = readStreamMsg(connectionId, requestId, stream.content)    
                            
        usage = stream.response_metadata['usage']
        print('prompt_tokens: ', usage['prompt_tokens'])
        print('completion_tokens: ', usage['completion_tokens'])
        print('total_tokens: ', usage['total_tokens'])
        msg = stream.content        
        #print('msg: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
            
        sendErrorMessage(connectionId, requestId, err_msg)    
        raise Exception ("Not able to request to LLM")

    if debugMessageMode == 'true':  
        chat_history = ""
        for dialogue_turn in history:
            #print('type: ', dialogue_turn.type)
            #print('content: ', dialogue_turn.content)
            
            dialog = f"{dialogue_turn.type}: {dialogue_turn.content}\n"            
            chat_history = chat_history + dialog
                
        history_length = len(chat_history)
        print('chat_history length: ', history_length)
        
        token_counter_history = 0
        if chat_history:
            token_counter_history = chat.get_num_tokens(chat_history)
            print('token_size of history: ', token_counter_history)
        
        end_time_for_inference = time.time()
        time_for_inference = end_time_for_inference - start_time_for_inference
    
    memory_chain.chat_memory.add_user_message(query)
    memory_chain.chat_memory.add_ai_message(msg)
    
    return msg

def store_document_for_faiss(docs, vectorstore_faiss):
    print('store document into faiss')    
    vectorstore_faiss.add_documents(docs)       
    print('uploaded into faiss')
    
def is_not_exist(index_name):    
    if os_client.indices.exists(index_name):        
        print('use exist index: ', index_name)    
        return False
    else:
        print('no index: ', index_name)
        return True    

def create_nori_index(category):
    index_body = {
        'settings': {
            'analysis': {
                'analyzer': {
                    'my_analyzer': {
                        'char_filter': ['html_strip'], 
                        'tokenizer': 'nori',
                        'filter': ['nori_number','lowercase','trim','my_nori_part_of_speech'],
                        'type': 'custom'
                    }
                },
                'tokenizer': {
                    'nori': {
                        'decompound_mode': 'mixed',
                        'discard_punctuation': 'true',
                        'type': 'nori_tokenizer'
                    }
                },
                "filter": {
                    "my_nori_part_of_speech": {
                        "type": "nori_part_of_speech",
                        "stoptags": [
                                "E", "IC", "J", "MAG", "MAJ",
                                "MM", "SP", "SSC", "SSO", "SC",
                                "SE", "XPN", "XSA", "XSN", "XSV",
                                "UNA", "NA", "VSV"
                        ]
                    }
                }
            },
            'index': {
                'knn': True,
                'knn.space_type': 'cosinesimil'  # Example space type
            }
        },
        'mappings': {
            'properties': {
                'metadata': {
                    'properties': {
                        'source' : {'type': 'keyword'},                    
                        'last_updated': {'type': 'date'},
                        'project': {'type': 'keyword'},
                        'seq_num': {'type': 'long'},
                        'title': {'type': 'text'},  # For full-text search
                        'url': {'type': 'text'},  # For full-text search
                    }
                },            
                'text': {
                    'analyzer': 'my_analyzer',
                    'search_analyzer': 'my_analyzer',
                    'type': 'text'
                },
                'vector_field': {
                    'type': 'knn_vector',
                    'dimension': 1536  # Replace with your vector dimension
                }
            }
        }
    }
    
    index_name = 'idx-rag-'+category
    
    if(is_not_exist(index_name)):
        try: # create index
            response = os_client.indices.create(
                index_name,
                body=index_body
            )
            print('index was created with nori plugin:', response)
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)                
            #raise Exception ("Not able to create the index")
            
def delete_document_if_exist(vectorstore, metadata_key):
    try: 
        s3r = boto3.resource("s3")
        bucket = s3r.Bucket(s3_bucket)
        objs = list(bucket.objects.filter(Prefix=metadata_key))
        print('objs: ', objs)
        
        if(len(objs)>0):
            doc = s3r.Object(s3_bucket, metadata_key)
            meta = doc.get()['Body'].read().decode('utf-8')
            print('meta: ', meta)
            
            ids = json.loads(meta)['ids']
            print('ids: ', ids)
            
            result = vectorstore.delete(ids)
            print('result: ', result)        
        else:
            print('no meta file: ', metadata_key)
            
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
        raise Exception ("Not able to create meta file")

if enableNoriPlugin == 'true':
    create_nori_index(category='py')
    create_nori_index(category='js')    
        
def store_document_for_opensearch(bedrock_embedding, docs, key, category):
    index_name = 'idx-rag-'+category
    vectorstore = OpenSearchVectorSearch(
        index_name=index_name,  
        is_aoss = False,
        #engine="faiss",  # default: nmslib
        embedding_function = bedrock_embedding,
        opensearch_url = opensearch_url,
        http_auth=(opensearch_account, opensearch_passwd),
    )
    
    objectName = (key[key.find(s3_prefix)+len(s3_prefix)+1:len(key)])
    print('objectName: ', objectName)    
    metadata_key = meta_prefix+objectName+'.metadata.json'
    print('meta file name: ', metadata_key)    
    delete_document_if_exist(vectorstore, metadata_key)
    
    try:        
        response = vectorstore.add_documents(docs, bulk_size = 2000)
        print('response of adding documents: ', response)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                
        #raise Exception ("Not able to request to LLM")

    print('uploaded into opensearch')
    
    return response
    
# load a code file from s3
def load_code(file_type, s3_file_name):
    s3r = boto3.resource("s3")
    doc = s3r.Object(s3_bucket, s3_prefix+'/'+s3_file_name)
    
    if file_type == 'py':        
        contents = doc.get()['Body'].read().decode('utf-8')
        separators = ["\ndef "]
        # print('contents: ', contents)
    elif file_type == 'js':
        contents = doc.get()['Body'].read().decode('utf-8')
        separators = ["\nfunction ", "\nexports.handler "]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=0,
        separators=separators,
        length_function = len,
    ) 

    texts = text_splitter.split_text(contents) 
    
    for i, text in enumerate(texts):
        print(f"Chunk #{i}: {text}")
                
    return texts

def summary_of_code(chat, code, mode):
    if mode == 'py':
        system = (
            "다음의 <article> tag에는 python code가 있습니다. code의 전반적인 목적에 대해 설명하고, 각 함수의 기능과 역할을 자세하게 한국어 500자 이내로 설명하세요."
        )
    elif mode == 'js':
        system = (
            "다음의 <article> tag에는 node.js code가 있습니다. code의 전반적인 목적에 대해 설명하고, 각 함수의 기능과 역할을 자세하게 한국어 500자 이내로 설명하세요."
        )
    else:
        system = (
            "다음의 <article> tag에는 code가 있습니다. code의 전반적인 목적에 대해 설명하고, 각 함수의 기능과 역할을 자세하게 한국어 500자 이내로 설명하세요."
        )
    
    human = "<article>{code}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    print('prompt: ', prompt)
    
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "code": code
            }
        )
        
        summary = result.content
        print('result of code summarization: ', summary)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return summary

def summarize_process_for_relevent_code(conn, chat, code, object, file_type, bedrock_region):
    try: 
        if code.find('\ndef ') != -1:
            start = code.find('\ndef ')
            end = code.find(':')   
        elif code.find('\nfunction ') != -1:
            start = code.find('\nfunction ')
            end = code.find('(')   
        elif code.find('\nexports.') != -1:
            start = code.find('\nexports.')
            end = code.find(' =')         
        else:
            start = -1
            end = -1
              
        print('code: ', code)                             
        print(f'start: {start}, end: {end}')
                    
        doc = ""    
        if start != -1:      
            function_name = code[start+1:end]
            # print('function_name: ', function_name)
                            
            summary = summary_of_code(chat, code, file_type)
            print(f"summary ({bedrock_region}): {summary}")
            
            #print('first line summary: ', summary[:len(function_name)])
            #print('function name: ', function_name)            
            if summary[:len(function_name)]==function_name:
                summary = summary[summary.find('\n')+1:len(summary)]

            doc = Document(
                page_content=summary,
                metadata={
                    'name': object,
                    'uri': path+doc_prefix+parse.quote(object),
                    'code': code,
                    'function_name': function_name
                }
            )            
                        
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)       
        # raise Exception (f"Not able to summarize: {doc}")               
    
    conn.send(doc)    
    conn.close()

def summarize_relevant_codes_using_parallel_processing(codes, object, file_type):
    selected_LLM = 0
    relevant_codes = []    
    processes = []
    parent_connections = []
    for code in codes:
        parent_conn, child_conn = Pipe()
        parent_connections.append(parent_conn)
            
        chat = get_chat(profile_of_LLMs, selected_LLM)
        bedrock_region = profile_of_LLMs[selected_LLM]['bedrock_region']

        process = Process(target=summarize_process_for_relevent_code, args=(child_conn, chat, code, object, file_type, bedrock_region))
        processes.append(process)

        selected_LLM = selected_LLM + 1
        if selected_LLM == len(profile_of_LLMs):
            selected_LLM = 0

    for process in processes:
        process.start()
            
    for parent_conn in parent_connections:
        doc = parent_conn.recv()
        
        if doc:
            relevant_codes.append(doc)    

    for process in processes:
        process.join()
    
    return relevant_codes
    
def load_chat_history(userId, allowTime):
    dynamodb_client = boto3.client('dynamodb')

    response = dynamodb_client.query(
        TableName=callLogTableName,
        KeyConditionExpression='user_id = :userId AND request_time > :allowTime',
        ExpressionAttributeValues={
            ':userId': {'S': userId},
            ':allowTime': {'S': allowTime}
        }
    )
    # print('query result: ', response['Items'])

    for item in response['Items']:
        text = item['body']['S']
        msg = item['msg']['S']
        type = item['type']['S']

        if type == 'text':
            memory_chain.chat_memory.add_user_message(text)
            if len(msg) > MSG_LENGTH:
                memory_chain.chat_memory.add_ai_message(msg[:MSG_LENGTH])                          
            else:
                memory_chain.chat_memory.add_ai_message(msg) 
                
def getAllowTime():
    d = datetime.datetime.now() - datetime.timedelta(days = 2)
    timeStr = str(d)[0:19]
    print('allow time: ',timeStr)

    return timeStr

def isTyping(connectionId, requestId):    
    msg_proceeding = {
        'request_id': requestId,
        'msg': 'Proceeding...',
        'status': 'istyping'
    }
    #print('result: ', json.dumps(result))
    sendMessage(connectionId, msg_proceeding)

def readStreamMsg(connectionId, requestId, stream):
    msg = ""
    if stream:
        for event in stream:
            #print('event: ', event)
            msg = msg + event

            result = {
                'request_id': requestId,
                'msg': msg,
                'status': 'proceeding'
            }
            #print('result: ', json.dumps(result))
            sendMessage(connectionId, result)
    # print('msg: ', msg)
    return msg

def priority_search(query, relevant_codes, bedrock_embedding):
    excerpts = []
    for i, doc in enumerate(relevant_codes):
        # print('doc: ', doc)
        content = doc['metadata']['excerpt']        
        excerpts.append(
            Document(
                page_content=content,
                metadata={
                    'name': doc['metadata']['title'],
                    'order':i,
                }
            )
        )  
    # print('excerpts: ', excerpts)

    embeddings = bedrock_embedding
    vectorstore_confidence = FAISS.from_documents(
        excerpts,  # documents
        embeddings  # embeddings
    )            
    rel_documents = vectorstore_confidence.similarity_search_with_score(
        query=query,
        k=top_k
    )

    docs = []
    for i, document in enumerate(rel_documents):
        print(f'## Document(priority_search) {i+1}: {document}')

        order = document[0].metadata['order']
        name = document[0].metadata['name']
        assessed_score = document[1]
        print(f"{order} {name}: {assessed_score}")

        relevant_codes[order]['assessed_score'] = int(assessed_score)

        if assessed_score < 300:
            docs.append(relevant_codes[order])    
    # print('selected docs: ', docs)

    return docs

def get_reference(docs):
    reference = "\n\nFrom\n"
    for i, doc in enumerate(docs):
        excerpt = doc['metadata']['excerpt'].replace('"','')
        code = doc['metadata']['code'].replace('"','')
        
        excerpt = excerpt.replace('\n','\\n')
        code = code.replace('\n','\\n')
        print('reference_doc: ', json.dumps(doc))
        
        if doc['rag_type'][:10] == 'opensearch':
            print(f'## Document(get_reference) {i+1}: {doc}')
                
            page = ""
            if "document_attributes" in doc['metadata']:
                if "_excerpt_page_number" in doc['metadata']['document_attributes']:
                    page = doc['metadata']['document_attributes']['_excerpt_page_number']
            uri = doc['metadata']['source']
            name = doc['metadata']['title']
            name = name[name.rfind('/')+1:len(name)]

            if page:                
                reference = reference + f"{i+1}. {page}page in <a href={uri} target=_blank>{name}</a>, {doc['rag_type']} ({doc['assessed_score']}), <a href=\"#\" onClick=\"alert(`{excerpt}`)\">코드설명</a>, <a href=\"#\" onClick=\"alert(`{code}`)\">관련코드</a>\n"
            else:
                reference = reference + f"{i+1}. <a href={uri} target=_blank>{name}</a>, {doc['rag_type']} ({doc['assessed_score']}), <a href=\"#\" onClick=\"alert(`{excerpt}`)\">코드설명</a>, <a href=\"#\" onClick=\"alert(`{code}`)\">관련코드</a>\n"
                            
    return reference

def checkDupulication(relevant_codes, doc_info):
    for doc in relevant_codes:
        if doc['metadata']['excerpt'] == doc_info['metadata']['excerpt']:
            return True
    return False

def retrieve_from_vectorstore(vectorstore_opensearch, query, top_k, rag_type):
    print(f"query: {query} ({rag_type})")
    relevant_codes = []
        
    if rag_type == 'opensearch':
        # Vector Search
        relevant_documents = vectorstore_opensearch.similarity_search_with_score(
            query = query,
            k = top_k,
        )
        print('relevant_documents: ', relevant_documents)

        for i, document in enumerate(relevant_documents):
            print(f'## Document(opensearch-vector) {i+1}: {document}')

            name = document[0].metadata['name']
            print('metadata: ', document[0].metadata)

            page = ""
            if "page" in document[0].metadata:
                page = document[0].metadata['page']
            uri = ""
            if "uri" in document[0].metadata:
                uri = document[0].metadata['uri']

            excerpt = document[0].page_content
            confidence = str(document[1])
            assessed_score = str(document[1])
            
            code = ""
            if "code" in document[0].metadata:
                code = document[0].metadata['code']
                
            function_name = ""
            if "function_name" in document[0].metadata:
                function_name = document[0].metadata['function_name']

            if page:
                print('page: ', page)
                doc_info = {
                    "rag_type": 'opensearch-vector',
                    "confidence": confidence,
                    "metadata": {
                        "source": uri,
                        "title": name,
                        "excerpt": excerpt,
                        "document_attributes": {
                            "_excerpt_page_number": page
                        },
                        "code": code,
                        "function_name": function_name
                    },
                    "assessed_score": assessed_score,
                }
            else:
                doc_info = {
                    "rag_type": 'opensearch-vector',
                    "confidence": confidence,
                    "metadata": {
                        "source": uri,
                        "title": name,
                        "excerpt": excerpt,
                        "code": code,
                        "function_name": function_name
                    },
                    "assessed_score": assessed_score,
                }
            relevant_codes.append(doc_info)
    
        # Lexical Search (keyword)
        min_match = 0
        if enableNoriPlugin == 'true':
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "match": {
                                    "text": {
                                        "query": query,
                                        "minimum_should_match": f'{min_match}%',
                                        "operator":  "or"
                                    }
                                }
                            },
                        ],
                        "filter": [
                        ]
                    }
                }
            }

            response = os_client.search(
                body=query,
                index="idx-*", # all
            )
            # print('lexical query result: ', json.dumps(response))
            
            for i, document in enumerate(response['hits']['hits']):
                if i>top_k: 
                    break
                
                excerpt = document['_source']['text']
                print(f'## Document(opensearch-keyward) {i+1}: {excerpt}')

                name = document['_source']['metadata']['name']
                print('name: ', name)

                page = ""
                if "page" in document['_source']['metadata']:
                    page = document['_source']['metadata']['page']
                
                uri = ""
                if "uri" in document['_source']['metadata']:
                    uri = document['_source']['metadata']['uri']
                print('uri: ', uri)

                confidence = str(document['_score'])
                assessed_score = ""
                
                code = ""
                if "code" in document['_source']['metadata']:
                    code = document['_source']['metadata']['code']
                
                function_name = ""
                if "function_name" in document['_source']['metadata']:
                    function_name = document['_source']['metadata']['function_name']

                if page:
                    print('page: ', page)
                    doc_info = {
                        "rag_type": 'opensearch-keyward',
                        #"api_type": api_type,
                        "confidence": confidence,
                        "metadata": {
                            #"type": query_result_type,
                            #"document_id": document_id,
                            "source": uri,
                            "title": name,
                            "excerpt": excerpt,
                            "document_attributes": {
                                "_excerpt_page_number": page
                            },
                            "code": code,
                            "function_name": function_name
                        },
                        #"query_id": query_id,
                        #"feedback_token": feedback_token
                        "assessed_score": assessed_score,
                    }
                else: 
                    doc_info = {
                        "rag_type": 'opensearch-keyward',
                        #"api_type": api_type,
                        "confidence": confidence,
                        "metadata": {
                            #"type": query_result_type,
                            #"document_id": document_id,
                            "source": uri,
                            "title": name,
                            "excerpt": excerpt,
                            "code": code,
                            "function_name": function_name
                        },
                        #"query_id": query_id,
                        #"feedback_token": feedback_token
                        "assessed_score": assessed_score,
                    }
                
                if checkDupulication(relevant_codes, doc_info) == False:
                    relevant_codes.append(doc_info)
                    
    return relevant_codes

def get_code_using_RAG(chat, text, conv_type, code_type, connectionId, requestId, bedrock_embedding):
    global time_for_rag, time_for_inference, time_for_priority_search, number_of_relevant_codes  # for debug
    time_for_rag = time_for_inference = time_for_priority_search = number_of_relevant_codes = 0
    
    category = code_type
    index_name =  f"idx-rag-{category}*"
    print('index: ', index_name)
        
    vectorstore_opensearch = OpenSearchVectorSearch(
        index_name = index_name,
        is_aoss = False,
        ef_search = 1024, # 512(default)
        m=48,
        embedding_function = bedrock_embedding,
        opensearch_url=opensearch_url,
        http_auth=(opensearch_account, opensearch_passwd), # http_auth=awsauth,
    )
        
    reference = ""
    start_time_for_rag = time.time()

    relevant_codes = [] 
    print('start RAG for question')

    rag_type = 'opensearch'
    relevant_codes = retrieve_from_vectorstore(vectorstore_opensearch=vectorstore_opensearch, query=text, top_k=top_k, rag_type=rag_type)
    print(f'relevant_codes ({rag_type}): '+json.dumps(relevant_codes))
    
    end_time_for_rag = time.time()
    time_for_rag = end_time_for_rag - start_time_for_rag
    print('processing time for RAG: ', time_for_rag)

    selected_relevant_codes = []
    if len(relevant_codes)>=1:
        selected_relevant_codes = priority_search(text, relevant_codes, bedrock_embedding)
        print('selected_relevant_codes: ', json.dumps(selected_relevant_codes))
    
    end_time_for_priority_search = time.time() 
    time_for_priority_search = end_time_for_priority_search - end_time_for_rag
    print('processing time for priority search: ', time_for_priority_search)
    number_of_relevant_codes = len(selected_relevant_codes)

    relevant_code = ""
    for document in selected_relevant_codes:
        if document['metadata']['code']:
            code = document['metadata']['code']
            relevant_code = relevant_code + code + "\n\n"            
    print('relevant_code: ', relevant_code)

    msg = generate_code(connectionId, requestId, chat, text, relevant_code, code_type)

    if len(selected_relevant_codes)>=1 and enableReference=='true':
        reference = get_reference(selected_relevant_codes)  

    end_time_for_inference = time.time()
    time_for_inference = end_time_for_inference - end_time_for_priority_search
    print('processing time for inference: ', time_for_inference)
    
    global relevant_length, token_counter_relevant_docs    
    if debugMessageMode=='true':   # extract chat history for debug
        relevant_length = len(relevant_code)
        token_counter_relevant_docs = chat.get_num_tokens(relevant_code)
    
    memory_chain.chat_memory.add_user_message(text)  # append new diaglog
    memory_chain.chat_memory.add_ai_message(msg)
    
    return msg, reference

def get_summary(chat, docs):    
    text = ""
    for doc in docs:
        text = text + doc
    
    if isKorean(text)==True:
        system = (
            "다음의 <article> tag안의 문장을 요약해서 500자 이내로 설명하세오."
        )
    else: 
        system = (
            "Here is pieces of article, contained in <article> tags. Write a concise summary within 500 characters."
        )
    
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    print('prompt: ', prompt)
    
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "text": text
            }
        )
        
        summary = result.content
        print('result of summarization: ', summary)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return summary

def generate_code(connectionId, requestId, chat, text, context, mode):
    if mode == 'py':    
        system = (
            """다음의 <context> tag안에는 질문과 관련된 python code가 있습니다. 이 코드를 참조하여 질문과 관련된 python 코드를 생성합니다. 
            
            <context>
            {context}
            </context>"""
        )
    elif mode == 'js':
        system = (
            """다음의 <context> tag안에는 질문과 관련된 node.js code가 있습니다. 이 코드를 참조하여 질문과 관련된 node.js 코드를 생성합니다. 
            
            <context>
            {context}
            </context>"""
        )
    
    human = "{text}"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    print('prompt: ', prompt)
    
    chain = prompt | chat    
    try: 
        isTyping(connectionId, requestId)  
        stream = chain.invoke(
            {
                "context": context,
                "text": text
            }
        )
        
        geenerated_code = readStreamMsg(connectionId, requestId, stream.content)
                              
        geenerated_code = stream.content        
        print('result of code generation: ', geenerated_code)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return geenerated_code

def summary_of_code(chat, code, mode):
    if mode == 'py':
        system = (
            "다음의 <article> tag에는 python code가 있습니다. code의 전반적인 목적에 대해 설명하고, 각 함수의 기능과 역할을 자세하게 한국어 500자 이내로 설명하세요."
        )
    elif mode == 'js':
        system = (
            "다음의 <article> tag에는 node.js code가 있습니다. code의 전반적인 목적에 대해 설명하고, 각 함수의 기능과 역할을 자세하게 한국어 500자 이내로 설명하세요."
        )
    else:
        system = (
            "다음의 <article> tag에는 code가 있습니다. code의 전반적인 목적에 대해 설명하고, 각 함수의 기능과 역할을 자세하게 한국어 500자 이내로 설명하세요."
        )
    
    human = "<article>{code}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    print('prompt: ', prompt)
    
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "code": code
            }
        )
        
        summary = result.content
        print('result of code summarization: ', summary)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                    
        raise Exception ("Not able to request to LLM")
    
    return summary

def create_metadata(bucket, key, meta_prefix, s3_prefix, uri, category, documentId, ids):
    title = key
    timestamp = int(time.time())

    metadata = {
        "Attributes": {
            "_category": category,
            "_source_uri": uri,
            "_version": str(timestamp),
            "_language_code": "ko"
        },
        "Title": title,
        "DocumentId": documentId,      
        "ids": ids  
    }
    print('metadata: ', metadata)
    
    #objectName = (key[key.find(s3_prefix)+len(s3_prefix)+1:len(key)]).upper()
    objectName = (key[key.find(s3_prefix)+len(s3_prefix)+1:len(key)])
    print('objectName: ', objectName)

    client = boto3.client('s3')
    try: 
        client.put_object(
            Body=json.dumps(metadata), 
            Bucket=bucket, 
            Key=meta_prefix+objectName+'.metadata.json' 
        )
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
        raise Exception ("Not able to create meta file")

def getResponse(connectionId, jsonBody):
    userId  = jsonBody['user_id']
    # print('userId: ', userId)
    requestId  = jsonBody['request_id']
    # print('requestId: ', requestId)
    requestTime  = jsonBody['request_time']
    # print('requestTime: ', requestTime)
    type  = jsonBody['type']
    # print('type: ', type)
    body = jsonBody['body']
    # print('body: ', body)
    conv_type = jsonBody['conv_type']  # conversation type
    print('Conversation Type: ', conv_type)
    code_type = jsonBody['code_type']  # code type
    print('code Type: ', code_type)
    
    object = body
    file_type = object[object.rfind('.')+1:len(object)]
    print('file_type: ', file_type)
    
    global map_chain, memory_chain, isReady, selected_LLM, enableReference

    # Multi-LLM
    profile = profile_of_LLMs[selected_LLM]
    bedrock_region =  profile['bedrock_region']
    modelId = profile['model_id']
    print(f'selected_LLM: {selected_LLM}, bedrock_region: {bedrock_region}, modelId: {modelId}')
    # print('profile: ', profile)

    # langchain for bedrock
    chat = get_chat(profile_of_LLMs, selected_LLM)    
    bedrock_embedding = get_embedding(profile_of_LLMs, selected_LLM)

    # create memory
    if userId in map_chain:  
        print('memory exist. reuse it!')        
        memory_chain = map_chain[userId]
    else: 
        print('memory does not exist. create new one!')        
        memory_chain = ConversationBufferWindowMemory(memory_key="chat_history", output_key='answer', return_messages=True, k=10)
        map_chain[userId] = memory_chain

        allowTime = getAllowTime()
        load_chat_history(userId, allowTime)

    start = int(time.time())    

    msg = ""
    reference = ""
    if type == 'text' and body[:11] == 'list models':
        bedrock_client = boto3.client(
            service_name='bedrock',
            region_name=bedrock_region,
        )
        modelInfo = bedrock_client.list_foundation_models()    
        print('models: ', modelInfo)

        msg = f"The list of models: \n"
        lists = modelInfo['modelSummaries']
        
        for model in lists:
            msg += f"{model['modelId']}\n"
        
        msg += f"current model: {modelId}"
        print('model lists: ', msg)          

        sendResultMessage(connectionId, requestId, msg)  
    else:             
        if type == 'text':
            text = body
            print('query: ', text)

            querySize = len(text)
            textCount = len(text.split())
            print(f"query size: {querySize}, words: {textCount}")

            if text == 'enableReference':
                enableReference = 'true'
                msg  = "Referece is enabled"
            elif text == 'disableReference':
                enableReference = 'false'
                msg  = "Reference is disabled"            
            elif text == 'clearMemory':
                memory_chain.clear()
                map_chain[userId] = memory_chain
                    
                print('initiate the chat memory!')
                msg  = "The chat memory was intialized in this session."
            else:        
                if conv_type == 'normal':      # normal
                    msg = general_conversation(connectionId, requestId, chat, text)    

                elif conv_type == 'qa':   # RAG
                    msg, reference = get_code_using_RAG(chat, text, conv_type, code_type, connectionId, requestId, bedrock_embedding)     
                
                # token counter
                if debugMessageMode=='true':
                    token_counter_input = chat.get_num_tokens(text)
                    token_counter_output = chat.get_num_tokens(msg)
                    print(f"token_counter: question: {token_counter_input}, answer: {token_counter_output}")
                
        elif type == 'document':
            isTyping(connectionId, requestId)
            
            ids = []            
            if file_type == 'py' or file_type == 'js':
                codes = load_code(file_type, object)  # number of functions in the code
                print('code: ', codes)
                
                docs = []
                msg = ""
                                
                if enableParallelSummay=='true':
                    docs = summarize_relevant_codes_using_parallel_processing(codes, object, file_type)
                    
                else:
                    for code in codes:
                        if code.find('\ndef ') != -1:
                            start = code.find('\ndef ')
                            end = code.find(':')   
                        elif code.find('\nfunction ') != -1:
                            start = code.find('\nfunction ')
                            end = code.find('(')   
                        elif code.find('\nexports.') != -1:
                            start = code.find('\nexports.')
                            end = code.find(' =')         
                        else:
                            start = -1
                            end = -1
                            
                        print('code: ', code)                             
                        print(f'start: {start}, end: {end}')
                        
                        if start != -1:      
                            function_name = code[start+1:end]
                            # print('function_name: ', function_name)
                                            
                            summary = summary_of_code(chat, code, file_type)
                                
                            if summary[:len(function_name)]==function_name:
                                summary = summary[summary.find('\n')+1:len(summary)]
                                                                            
                            docs.append(
                                Document(
                                    page_content=summary,
                                    metadata={
                                        'name': object,
                                        # 'page':i+1,
                                        'uri': path+doc_prefix+parse.quote(object),
                                        'code': code,
                                        'function_name': function_name
                                    }
                                )
                            )      
                
                msg = ""
                for doc in docs:   
                    function_name = doc.metadata['function_name']
                    summary = doc.page_content
                    msg = msg + f'{function_name}:\n{summary}\n\n'
                                 
                print('docs size: ', len(docs))
                if len(docs)>=1:
                    print('docs[0]: ', docs[0])         
                
                msg = msg.replace('\n\n\n', '\n\n')      
            else:
                # msg = "uploaded file: "+object
                msg = f"{file_type} is not supported"
                                
            if conv_type == 'qa':
                start_time = time.time()
                
                category = file_type
                key = doc_prefix+object
                    
                ids = store_document_for_opensearch(bedrock_embedding, docs, key, category)

                print('processing time: ', str(time.time() - start_time))
                
                documentId = category + "-" + key
                documentId = documentId.replace(' ', '_') # remove spaces
                documentId = documentId.replace(',', '_') # remove commas # not allowed: [ " * \\ < | , > / ? ]
                documentId = documentId.replace('/', '_') # remove slash
                documentId = documentId.lower() # change to lowercase
                print('documentId: ', documentId)
            
            create_metadata(bucket=s3_bucket, key=key, meta_prefix=meta_prefix, s3_prefix=s3_prefix, uri=path+parse.quote(key), category=category, documentId=documentId, ids=ids)
        
        msg = msg.replace(" ","&nbsp;")                  
        sendResultMessage(connectionId, requestId, msg+reference)
        # print('msg+reference: ', msg+reference)
                         
        if reference: # Summarize the generated code 
            generated_code = msg[msg.find('<result>')+9:len(msg)-10]
            generated_code_summary = summary_of_code(chat, generated_code, file_type)    
            msg += f'\n\n[생성된 코드 설명]\n{generated_code_summary}'
            msg = msg.replace('\n\n\n', '\n\n') 
            
            sendResultMessage(connectionId, requestId, msg+reference)

        elapsed_time = str(time.time() - start)
        print("total run time(sec): ", elapsed_time)

        item = {    # save dialog
            'user_id': {'S':userId},
            'request_id': {'S':requestId},
            'request_time': {'S':requestTime},
            'type': {'S':type},
            'body': {'S':body},
            'msg': {'S':msg+reference}
        }
        client = boto3.client('dynamodb')
        try:
            resp =  client.put_item(TableName=callLogTableName, Item=item)
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)
            raise Exception ("Not able to write into dynamodb")        
        #print('resp, ', resp)

    if selected_LLM >= len(profile_of_LLMs)-1:
        selected_LLM = 0
    else:
        selected_LLM = selected_LLM + 1

    return msg, reference

def lambda_handler(event, context):
    # print('event: ', event)
    
    msg = ""
    if event['requestContext']: 
        connectionId = event['requestContext']['connectionId']        
        routeKey = event['requestContext']['routeKey']
        
        if routeKey == '$connect':
            print('connected!')
        elif routeKey == '$disconnect':
            print('disconnected!')
        else:
            body = event.get("body", "")
            #print("data[0:8]: ", body[0:8])
            if body[0:8] == "__ping__":
                # print("keep alive!")                
                sendMessage(connectionId, "__pong__")
            else:
                print('connectionId: ', connectionId)
                print('routeKey: ', routeKey)
        
                jsonBody = json.loads(body)
                print('request body: ', json.dumps(jsonBody))

                requestId  = jsonBody['request_id']
                try:
                    msg, reference = getResponse(connectionId, jsonBody)

                    print('msg+reference: ', msg+reference)
                                        
                except Exception:
                    err_msg = traceback.format_exc()
                    print('err_msg: ', err_msg)

                    sendErrorMessage(connectionId, requestId, err_msg)    
                    raise Exception ("Not able to send a message")

    return {
        'statusCode': 200
    }
