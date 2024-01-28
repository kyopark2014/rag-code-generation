# RAG를 이용한 Code Generation

RAG를 이용해 코드를 검색하고 이때 얻어진 코드를 이용하여 목적에 맞는 코드를 생성합니다.

### 코드 요약

업로드한 코드를 함수별로 나누는 chunking을 수행합니다.

### Chunk 별로 코드 요약

코드는 함수별로 Chunk 되었으므로 각 함수에 대한 요약을 수행합니다. 수행속도를 위해 Multi-LLM을 활용합니다.

### 코드 요약을 OpenSearch에 등록

RAG로 OpenSearch를 이용합니다. 각 Chunk는 함수 1개씩을 의미하므로 코드 요약과 원본 코드를 아래와 같이 metadata에 저장합니다.

### 관련된 코드를 RAG로 조회

사용자가 원하는 코드를 검색하면 RAG로 조회합니다. 이때, OpenSearch로 vector와 lexical search를 하여 두개의 결과를 병합하고, priority search를 통해 관련된 코드의 우선 순위를 조정합니다.

본 게시글에서는 관련된 코드를 검색할때 Zero shot을 이용하므로, "OpenSearch에서 Knowledge Store 생성하기"와 같이 구현하려는 Code에 대한 명확한 지시를 내려야 좀 더 정확한 결과를 얻을 수 있습니다. 만약, 대화이력을 고려하여 코드를 생성하고자 한다면, [한영 동시 검색 및 인터넷 검색을 활용하여 RAG를 편리하게 활용하기](https://aws.amazon.com/ko/blogs/tech/rag-enhanced-searching/)와 같이 Prompt를 이용하여 새로운 질문(Revised Question)을 생성할 수 있습니다. 

### 관련된 문서를 가지고 context를 생성

Context에 관련된 문서를 넣어서 아래와 같은 prompt를 이용하여 질문에 맞는 코드를 생성합니다.

### 코드의 Reference 

코드를 사용할때 원본 코드, 참고한 코드 정보를 함께 보여주어서, 참조한 코드의 활용도를 높입니다.

## 결과

![result](https://github.com/kyopark2014/rag-code-generation/assets/52392004/1863643d-d263-408c-ae54-dfea3aa9eff5)
