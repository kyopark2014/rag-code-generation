# RAG를 이용한 Code Generation

[Amazon CodeWhisperer](https://docs.aws.amazon.com/codewhisperer/latest/userguide/what-is-cwspr.html)와 같은 Machine Learning 기반의 코드 생성 툴은 기업의 생산성 향상에 도움을 주고 있습니다. 하지만, 기업의 자산인 소스 코드들을 이러한 툴과 함께 활용하기 위하여 [Fine Tunining](https://docs.aws.amazon.com/ko_kr/sagemaker/latest/dg/jumpstart-fine-tune.html)을 하려면 비용도 고려해야 하고, 소스 코드들이 계속 업데이트 될 경우에 Fine Tuning 주기에 대한 부담이 있을 수 있습니다. 반면에 [RAG (Retrieval Augmented Generation)](https://docs.aws.amazon.com/ko_kr/sagemaker/latest/dg/jumpstart-foundation-models-customize-rag.html)은 [Amazon OpenSearch](https://docs.aws.amazon.com/ko_kr/opensearch-service/latest/developerguide/what-is.html)와 같은 검색 엔진을 활용하여 Fine Tuning과 유사한 기능을 제공할 수 있고, 일반적으로 업데이트나 비용면에서 Fine tuning 대비 유용합니다. 

본 게시글에서는 [LLM(Large Language Models)](https://aws.amazon.com/ko/what-is/large-language-model/)과 OpenSearch를 이용하여 RAG를 구성하고, 한국어로 된 질문(Query)으로 코드를 검색하고 이때 얻어진 관련된 코드들(Relevant Codes)을 이용하여 질문(Query)에 맞는 코드를 생성합니다. 질문과 관련된 코드를 찾기 위해서, LLM을 이용하여 함수(Function)에 대한 요약을 수행하고, 얻어진 요약과 원본 코드를 RAG에 등록하여 사용자의 질문이 있을 때에 관련된 코드들(Relevant Codes)를 검색하여 활용합니다. 코드에는 여러 개의 함수가 사용될 수 있으므로 코드 요약시에 걸리는 시간 지연이 증가할 수 있고, 여러 파일을 On-Demend에서 등록할 경우에 [요청과 Token 수의 제한](https://docs.aws.amazon.com/bedrock/latest/userguide/quotas.html)이 있으므로 [Multi-Region LLM](https://aws.amazon.com/ko/blogs/tech/multi-rag-and-multi-region-llm-for-chatbot/)을 활용합니다.

## Architecture 개요

전체적인 Architecture는 아래와 같습니다. 사용자는 [WebSocket 방식의 API Gateway](https://aws.amazon.com/ko/blogs/tech/stream-chatbot-for-amazon-bedrock/)를 이용하여 메시지를 주고 받습니다. [서버리스(Serverless)](https://docs.aws.amazon.com/ko_kr/serverless-application-model/latest/developerguide/what-is-concepts.html)인 [AWS Lambda](https://docs.aws.amazon.com/lambda/latest/dg/welcome.html)를 이용하여 RAG의 Knowledge Store로 부터 관련된 코드를 검색하고, [Prompt](https://docs.aws.amazon.com/ko_kr/bedrock/latest/userguide/what-is-a-prompt.html)를 이용해 질문(Query)에 맞는 코드를 생성합니다. Amazon OpenSearch는 빠르고 성능이 좋은 검색 엔진으로서, 관련된 코드를 빠르고 효과적으로 검색할 수 있습니다. 여기서는 검색의 정확도를 높이기 위해 OpenSearch의 [Vector 검색](https://opensearch.org/platform/search/vector-database.html)과 함께 [한국어를 지원하는 Nori 분석기](https://aws.amazon.com/ko/blogs/tech/amazon-opensearch-service-korean-nori-plugin-for-analysis/)를 이용하여 Lexical 검색을 수행합니다. 검색된 결과들은 관련도에 따라 정렬할 필요가 있으므로 [Faiss의 similarity search](https://aws.amazon.com/ko/blogs/tech/rag-enhanced-searching/)를 이용하여 Priority Search를 수행합니다. 여기서 사용하는 Faiss는 Lambda의 process와 memory를 공유하므로 별도로 비용이 발생하지 않으며 정량적인 관련도를 얻을 수 있습니다. 잘문과 관련된 코드를 한국어로 검색하기 위하여 RAG에 저장되는 소스 코드들은 함수(Function) 단위로 Chunking 된 후에 LLM을 이용하여 요약(Summarization)됩니다. 하나의 소스 파일은 여러 개의 함수들로 구성될 수 있으므로,  [Multi-Region LLM](https://aws.amazon.com/ko/blogs/tech/multi-rag-and-multi-region-llm-for-chatbot/)을 이용하여 요약에 필요한 지연 속도를 개선합니다. 또한, 인프라를 배포하고 관리하는 것은 [AWS CDK](https://docs.aws.amazon.com/cdk/v2/guide/home.html)을 이용합니다.

<img src="https://github.com/kyopark2014/rag-code-generation/assets/52392004/3f5e891c-3cbf-44d5-b337-82229e0a10f9" width="900">

전체적인 동작을 위한 Sequence Diagram은 아래와 같습니다. 

단계 1: 사용자가 파일을 업로드하려고 하면 [Presigned URL](https://docs.aws.amazon.com/ko_kr/AmazonS3/latest/userguide/PresignedUrlUploadObject.html)을 이용하여 [Amazon S3](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html)에 업로드합니다. 

단계 2: 파일 업로드 후에 type이 Document인 메시지를 [Amazon API Gateway]를 경유하여 Lambda에 보냅니다. Lambda는 S3에서 소스 파일을 로드(Load)한 후에, 함수 단위로 Chunking을 수행합니다. 

단계 3: 각 함수(Function)의 기능을 한국어로 설명할 수 있도록 LLM을 이용하여 코드를 요약(Summarization)합니다. 이때, 요약하는 속도를 향상시키기 위하여 아래처럼 4개 Region의 LLM을 활용하여 요약 작업을 수행합니다.

단계 4: 각 함수(Function)에 대한 요약과 원본 코드, 파일 경로를 메타데이터로 가지는 [Document](https://opensearch.org/docs/latest/)를 만들어서 OpenSearch에 저장하고, 채팅창에는 소스 코드에 대한 요약 결과를 보여줍니다.

단계 5: 사용자가 코드를 생성하기 위하여 질문(Query)을 입력하면, OpenSearch로 Vector/Lexical Search를 수행하여 관련된 코드들을 수집합니다.

단계 6: 관련된 코드들은 Priority Search를 이용하여 관련도가 높은 순서로 정렬한 다음에 일정 수준의 관련도를 가지는 관련 코드로 LLM에서 활용할 수 있는 Context를 만듭니다.

단계 7: 관련된 코드의 조합인 Context와 사용자의 질문(Query)을 LLM에 전달하여 코드를 생성합니다.

![image](https://github.com/kyopark2014/rag-code-generation/assets/52392004/2cf62fbc-e8dd-4704-993f-864f9ee3dbc1)


## 주요 시스템 구성

### 코드 요약

본 게시글에서는 함수 단위로 코드를 요약하여 RAG에서 활용하고자 합니다. 하나의 소스 파일은 여러 개의 함수(Function)을 가질 수 있으므로, S3로 부터 소스 파일을 읽어 들인 후에 함수(Function) 단위로 분리하기 위하여 Chunking을 수행합니다. 여기서는 chunk_size를 50으로 하고, seperator로 "\ndef "로 설정하였으므로, 50자 이상의 함수로 분리 할 수 있습니다.

```python
def load_code(file_type, s3_file_name):
    s3r = boto3.resource("s3")
    doc = s3r.Object(s3_bucket, s3_prefix+'/'+s3_file_name)
    
    if file_type == 'py':        
        contents = doc.get()['Body'].read().decode('utf-8')
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=0,
        separators=["\ndef "],
        length_function = len,
    ) 

    texts = text_splitter.split_text(contents) 
                
    return texts
```

### 함수 별로 코드 요약

 함수별로 Chunk 되었으므로 LLM을 이용하여 요약을 수행합니다. 이때 수행속도를 위해 Multi-LLM과 Multi Thread를 활용합니다. 여기서는 여러 리전의 LLM을 활용하므로 LLM profile에 여러 리전에 대한 정보와 모델을 설정합니다. Chunk된 문서는 파일 이름(name), 다운로드 경로(uri), 원본 코드(code)와 각 함수의 이름(function_name)을 metadata로 가지고 있습니다. 
 
```python
 def summarize_relevant_codes_using_parallel_processing(codes, object):
    selected_LLM = 0
    relevant_codes = []    
    processes = []
    parent_connections = []
    for code in codes:
        parent_conn, child_conn = Pipe()
        parent_connections.append(parent_conn)
            
        llm = get_llm(profile_of_LLMs, selected_LLM)
        bedrock_region = profile_of_LLMs[selected_LLM]['bedrock_region']

        process = Process(target=summarize_process_for_relevent_code, args=(child_conn, llm, code, object, bedrock_region))
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

def summarize_process_for_relevent_code(conn, llm, code, object, bedrock_region):
    try: 
        start = code.find('\ndef ')
        end = code.find(':')                    
                    
        doc = ""    
        if start != -1:      
            function_name = code[start+1:end]
                            
            summary = summary_of_code(llm, code)
            print(f"summary ({bedrock_region}): {summary}")
            
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
    
    conn.send(doc)    
    conn.close()
```

코드를 요약하기 위해 Prompt를 활용합니다. 요약한 결과만을 추출하기 위하여 <result> tag를 활용하였고, 불필요한 줄바꿈은 아래와 같이 삭제하였습니다. 

```python
def summary_of_code(llm, code):
    PROMPT = """\n\nHuman: 다음의 <article> tag에는 python code가 있습니다. 각 함수의 기능과 역할을 자세하게 500자 이내로 설명하세요. 결과는 <result> tag를 붙여주세요.
           
    <article>
    {input}
    </article>
                        
    Assistant:"""
 
    try:
        summary = llm(PROMPT.format(input=code))
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)        
        raise Exception ("Not able to summary the message")
   
    summary = summary[summary.find('<result>')+9:len(summary)-10] # remove <result> tag
    
    summary = summary.replace('\n\n', '\n') 
    if summary[0] == '\n':
        summary = summary[1:len(summary)]
   
    return summary
```

### 코드 요약을 OpenSearch에 등록

RAG로 OpenSearch를 이용합니다. [2023년 10월에 한국어, 일본어, 중국어에 대한 새로운 언어 분석기 플러그인이 OpenSearch에 추가](https://aws.amazon.com/ko/about-aws/whats-new/2023/10/amazon-opensearch-four-language-analyzers/) 되었습니다. 이제 OpenSearch에서 한국어를 Nori 분석기를 이용하여 Lexical 검색을 이용하고 이를 이용해 RAG를 구현할 수 있습니다. 여기서는 OpenSearch에서 [Nori 플러그인을 이용한 한국어 분석](https://aws.amazon.com/ko/blogs/tech/amazon-opensearch-service-korean-nori-plugin-for-analysis/) 블로그와 [01_2_load_json_kr_docs_opensearch.ipynb](https://github.com/aws-samples/aws-ai-ml-workshop-kr/blob/master/genai/aws-gen-ai-kr/20_applications/02_qa_chatbot/01_preprocess_docs/01_2_load_json_kr_docs_opensearch.ipynb)를 참조하였습니다.

```python
if file_type == 'py':
    category = file_type
    key = doc_prefix+object
    documentId = category + "-" + key
    documentId = documentId.replace(' ', '_') 
    documentId = documentId.replace(',', '_') 
    documentId = documentId.replace('/', '_') 
    documentId = documentId.lower() 

    store_document_for_opensearch_with_nori(bedrock_embeddings, docs, documentId)

def store_document_for_opensearch_with_nori(bedrock_embeddings, docs, documentId):
    index_name = get_index_name(documentId)
    
    delete_index_if_exist(index_name)
    
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
                'knn.space_type': 'cosinesimil' 
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
                        'title': {'type': 'text'},  
                        'url': {'type': 'text'},  
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
    
    try: # create index
        response = os_client.indices.create(
            index_name,
            body=index_body
        )
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                

    try: # put the doucment
        vectorstore = OpenSearchVectorSearch(
            index_name=index_name,  
            is_aoss = False,
            embedding_function = bedrock_embeddings,
            opensearch_url = opensearch_url,
            http_auth=(opensearch_account, opensearch_passwd),
        )
        response = vectorstore.add_documents(docs, bulk_size = 2000)
        print('response of adding documents: ', response)
    except Exception:
        err_msg = traceback.format_exc()
        print('error message: ', err_msg)                
```

### RAG의 Knowledge Store를 이용하여 관련 Code 검색

사용자가 질문(Query)를 입력하면, RAG로 관련된 코드를 조회합니다. 이때, OpenSearch로 vector와 lexical search를 하여 두개의 결과를 병합하고, priority search를 통해 관련된 코드의 우선 순위를 조정합니다. 본 게시글에서는 관련된 코드를 검색할때 Zero shot을 이용하므로, 와 같이 구현하려는 코드에 대한 명확한 지시를 내려야 좀 더 정확한 결과를 얻을 수 있습니다. 만약, 대화이력을 고려하여 코드를 생성하고자 한다면, [한영 동시 검색 및 인터넷 검색을 활용하여 RAG를 편리하게 활용하기](https://aws.amazon.com/ko/blogs/tech/rag-enhanced-searching/)와 같이 Prompt를 이용하여 새로운 질문(Revised Question)을 생성할 수 있습니다. 

```python
def retrieve_from_vectorstore(query, top_k, rag_type):
    relevant_codes = []

    # Vector Search
    if rag_type == 'opensearch':
        relevant_documents = vectorstore_opensearch.similarity_search_with_score(
            query = query,
            k = top_k,
        )

        for i, document in enumerate(relevant_documents):
            name = document[0].metadata['name']
            uri = document[0].metadata['uri']
            excerpt = document[0].page_content
            confidence = str(document[1])
            assessed_score = str(document[1])            
            code = document[0].metadata['code']                
            function_name = document[0].metadata['function_name']

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
            relevant_codes.append(doc_info)
    
        # Lexical search (keyword)
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
                                        "operator":  "or",
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
            
            for i, document in enumerate(response['hits']['hits']):
                if i>top_k: 
                    break
                
                excerpt = document['_source']['text']
                name = document['_source']['metadata']['name']
                uri = document['_source']['metadata']['uri']
                confidence = str(document['_score'])
                assessed_score = ""                
                code = document['_source']['metadata']['code']                
                function_name = document['_source']['metadata']['function_name']

                doc_info = {
                    "rag_type": 'opensearch-keyward',
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
                if checkDupulication(relevant_codes, doc_info) == False:
                    relevant_codes.append(doc_info)
                    
    return relevant_codes
```


### 관련된 Code를 가지고 Context를 생성

Context에 관련된 문서를 넣어서 아래와 같은 prompt를 이용하여 질문에 맞는 코드를 생성합니다.

```python
selected_relevant_codes = []
if len(relevant_codes) >= 1:
    selected_relevant_codes = priority_search(text, relevant_codes, bedrock_embeddings)

relevant_code = ""
for document in selected_relevant_codes:
    if document['metadata']['code']:
        code = document['metadata']['code']
        relevant_code = relevant_code + code + "\n\n"

try:
    isTyping(connectionId, requestId)
    stream = llm(PROMPT.format(context = relevant_code, question = text))
    msg = readStreamMsg(connectionId, requestId, stream)                    
except Exception:
    err_msg = traceback.format_exc()
    print('error message: ', err_msg)
```

Faiss의 Similarity Search를 이용하여 관련도 기준으로 정렬합니다. 관련된 문서
       
```python
def priority_search(query, relevant_codes, bedrock_embeddings):
    excerpts = []
    for i, doc in enumerate(relevant_codes):
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

    embeddings = bedrock_embeddings
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

        relevant_codes[order]['assessed_score'] = int(assessed_score)

        if assessed_score < 400:
            docs.append(relevant_codes[order])    
    return docs
```

### Code의 Reference 

코드를 사용할때 원본 코드, 참고한 코드 정보를 함께 보여주어서, 참조한 코드의 활용도를 높입니다.

```python
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
```    

## 직접 실습 해보기

### 사전 준비 사항

이 솔루션을 사용하기 위해서는 사전에 아래와 같은 준비가 되어야 합니다.

- [AWS Account 생성](https://repost.aws/ko/knowledge-center/create-and-activate-aws-account)에 따라 계정을 준비합니다.

### CDK를 이용한 인프라 설치

본 실습에서는 Seoul 리전 (ap-northeast-2)을 사용합니다. [인프라 설치](./deployment.md)에 따라 CDK로 인프라 설치를 진행합니다. [CDK 구현 코드](./cdk-rag-chatbot-with-kendra/README.md)에서는 Typescript로 인프라를 정의하는 방법에 대해 상세히 설명하고 있습니다. 

## 실행결과

[lambda_function.py](https://github.com/kyopark2014/rag-code-generation/blob/main/lambda-chat-ws/lambda_function.py)을 다운로드 후에 채팅창 아래의 파일 아이콘을 선택하여 업로드합니다. lambda_function.py가 가지고 있는 함수들에 대한 요약을 보여줍니다.

![image](https://github.com/kyopark2014/rag-code-generation/assets/52392004/44b752de-f1fb-43e9-a6f6-e7ade65dbcb8)

채팅창에 아래와 같이 "OpenSearch에서 Knowledge Store 생성하기"라고 입력하고 결과를 확인합니다.

![result](https://github.com/kyopark2014/rag-code-generation/assets/52392004/1863643d-d263-408c-ae54-dfea3aa9eff5)


## 리소스 정리하기 

더이상 인프라를 사용하지 않는 경우에 아래처럼 모든 리소스를 삭제할 수 있습니다. 

1) [API Gateway Console](https://ap-northeast-2.console.aws.amazon.com/apigateway/main/apis?region=ap-northeast-2)로 접속하여 "api-chatbot-for-rag-code-generation", "api-rag-code-generation"을 삭제합니다.

2) [Cloud9 Console](https://ap-northeast-2.console.aws.amazon.com/cloud9control/home?region=ap-northeast-2#/)에 접속하여 아래의 명령어로 전체 삭제를 합니다.


```text
cd ~/environment/rag-code-generation/cdk-rag-code-generation/ && cdk destroy --all
```





## 결론

RAG를 이용하여 Code 생성하였습니다.


## 실습 코드 및 도움이 되는 참조 블로그

아래의 링크에서 실습 소스 파일 및 기계 학습(ML)과 관련된 자료를 확인하실 수 있습니다.

- [Amazon SageMaker JumpStart를 이용하여 Falcon Foundation Model기반의 Chatbot 만들기](https://aws.amazon.com/ko/blogs/tech/chatbot-based-on-falcon-fm/)
- [Amazon SageMaker JumpStart와 Vector Store를 이용하여 Llama 2로 Chatbot 만들기](https://aws.amazon.com/ko/blogs/tech/sagemaker-jumpstart-vector-store-llama2-chatbot/)
- [VARCO LLM과 Amazon OpenSearch를 이용하여 한국어 Chatbot 만들기](https://aws.amazon.com/ko/blogs/tech/korean-chatbot-using-varco-llm-and-opensearch/)
- [Amazon Bedrock을 이용하여 Stream 방식의 한국어 Chatbot 구현하기](https://aws.amazon.com/ko/blogs/tech/stream-chatbot-for-amazon-bedrock/)
- [Multi-RAG와 Multi-Region LLM로 한국어 Chatbot 만들기](https://aws.amazon.com/ko/blogs/tech/multi-rag-and-multi-region-llm-for-chatbot/)
- [한영 동시 검색 및 인터넷 검색을 활용하여 RAG를 편리하게 활용하기](https://aws.amazon.com/ko/blogs/tech/rag-enhanced-searching/)
- [Amazon Bedrock의 Claude와 Amazon Kendra로 향상된 RAG 사용하기](https://aws.amazon.com/ko/blogs/tech/bedrock-claude-kendra-rag/)

