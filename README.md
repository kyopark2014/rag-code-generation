# RAG를 이용한 Code Generation

[Amazon CodeWhisperer](https://docs.aws.amazon.com/codewhisperer/latest/userguide/what-is-cwspr.html)와 같은 Machine Learning 기반의 Code 생성을 돕는 툴들은 기업의 생산성 향상에 도움을 주고 있습니다. 하지만 기업의 자산인 Code들을 이러한 툴들과 함게 활용하기 위해 [Fine Tunining](https://docs.aws.amazon.com/ko_kr/sagemaker/latest/dg/jumpstart-fine-tune.html)을 하려면 비용도 고려해야하고 관련된 Code들이 계속 업데이트 될 경우에 새로운 Fine Tuning에 대한 부담이 있을 수 있습니다.반면에 [RAG (Retrieval Augmented Generation)](https://docs.aws.amazon.com/ko_kr/sagemaker/latest/dg/jumpstart-foundation-models-customize-rag.html)은 [LLM (Large Language Model)](https://aws.amazon.com/ko/what-is/large-language-model/)과 [Amazon OpenSearch](https://docs.aws.amazon.com/ko_kr/opensearch-service/latest/developerguide/what-is.html)과 같은 검색 엔진을 활용하여 Fine-Tunining과 유사한 기능을 제공할 수 있습니다. 

본 게시글에서는 LLM과 OpenSearch를 이용하여 RAG를 구성하고, 요청에 맞는 Code를 검색하고 이때 얻어진 관련된 Code를 이용하여 목적에 맞는 Code를 생성합니다. 






## Architecture 개요

전체적인 Architecture는 아래와 같습니다. 사용자는 [WebSocket 방식의 API Gateway](https://aws.amazon.com/ko/blogs/tech/stream-chatbot-for-amazon-bedrock/)를 이용하여 접속하고, 유지보수에 유리한 Amazon Lambda를 이용하여 RAG의 Knowledge Store로 부터 관련된 code를 검색하고, Prompt를 이용해 Code를 생성합니다. Amazon OpenSearch는 매우 빠르고 성능이 좋은 검색 엔진으로 다양한 Code를 빠르고 검색할 수 있습니다. 여기서는 OpenSearch의 [Vector 검색](https://opensearch.org/platform/search/vector-database.html)과 함께, 검색의 정확도를 높이기 위해 [한국어를 지원하는 Nori 분석기](https://aws.amazon.com/ko/blogs/tech/amazon-opensearch-service-korean-nori-plugin-for-analysis/)를 이용하여 Lexical 검색을 수행합니다. 잘문과 관련된 Code를 한국어로 검색하기 위하여 RAG에 저장되는 Reference Code들은 Function 단위로 Chunking된 후에 LLM을 이용하여 요약(Summerazation)됩니다. Function 별로 요약할때 속도를 개선하기 위하여 [Multi-Region LLM](https://aws.amazon.com/ko/blogs/tech/multi-rag-and-multi-region-llm-for-chatbot/)을 활용합니다. AWS에 이러한 인프라를 배포하고 관리하는것은 [AWS CDK](https://docs.aws.amazon.com/cdk/v2/guide/home.html)을 이용합니다.

<img src="https://github.com/kyopark2014/rag-code-generation/assets/52392004/3f5e891c-3cbf-44d5-b337-82229e0a10f9" width="900">

Sequence Diagram은 아래와 같습니다.

![image](https://github.com/kyopark2014/rag-code-generation/assets/52392004/2cf62fbc-e8dd-4704-993f-864f9ee3dbc1)

## 주요 시스템 구성

### Code 요약

업로드한 코드를 함수별로 나누는 chunking을 수행합니다.

### 함수 별로 코드 요약

 함수별로 Chunk 되었으므로 각 함수에 대한 요약을 수행합니다. 수행속도를 위해 Multi-LLM을 활용합니다.

### Code 요약을 OpenSearch에 등록

RAG로 OpenSearch를 이용합니다. 각 Chunk는 함수 1개씩을 의미하므로 코드 요약과 원본 코드를 아래와 같이 metadata에 저장합니다.

### RAG의 Knowledge Store를 이용하여 관련 Code 검색

사용자가 원하는 코드를 검색하면 RAG로 조회합니다. 이때, OpenSearch로 vector와 lexical search를 하여 두개의 결과를 병합하고, priority search를 통해 관련된 코드의 우선 순위를 조정합니다.

본 게시글에서는 관련된 코드를 검색할때 Zero shot을 이용하므로, 와 같이 구현하려는 Code에 대한 명확한 지시를 내려야 좀 더 정확한 결과를 얻을 수 있습니다. 만약, 대화이력을 고려하여 코드를 생성하고자 한다면, [한영 동시 검색 및 인터넷 검색을 활용하여 RAG를 편리하게 활용하기](https://aws.amazon.com/ko/blogs/tech/rag-enhanced-searching/)와 같이 Prompt를 이용하여 새로운 질문(Revised Question)을 생성할 수 있습니다. 

### 관련된 Code를 가지고 Context를 생성

Context에 관련된 문서를 넣어서 아래와 같은 prompt를 이용하여 질문에 맞는 코드를 생성합니다.

### Code의 Reference 

Code를 사용할때 원본 코드, 참고한 Code 정보를 함께 보여주어서, 참조한 코드의 활용도를 높입니다.

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

