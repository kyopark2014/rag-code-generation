# 인프라 설치하기

## Bedrock 사용 권한 설정하기

Tokyo(ap-northeast-1), Frankfurt(eu-central-1), Virginia(us-east-1), Oregon(us-west-2) 리전의 Bedrock을 사용합니다. [Model access - Tokyo](https://ap-northeast-1.console.aws.amazon.com/bedrock/home?region=ap-northeast-1#/modelaccess), [Model access - Frankfurt](https://eu-central-1.console.aws.amazon.com/bedrock/home?region=eu-central-1#/modelaccess), [Model access - Virginia](https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/modelaccess), [Model access - Oregon](https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/modelaccess)에 접속해서 [Edit]를 선택하여 모든 모델을 사용할 수 있도록 설정합니다. 특히 Anthropic Claude와 "Titan Embeddings G1 - Text"은 LLM 및 Vector Embedding을 위해서 반드시 사용이 가능하여야 합니다.



## CDK를 이용한 인프라 설치하기


여기서는 [Cloud9](https://aws.amazon.com/ko/cloud9/)에서 [AWS CDK](https://aws.amazon.com/ko/cdk/)를 이용하여 인프라를 설치합니다.

1) [Cloud9 Console](https://ap-northeast-2.console.aws.amazon.com/cloud9control/home?region=ap-northeast-2#/create)에 접속하여 [Create environment]-[Name]에서 “chatbot”으로 이름을 입력하고, EC2 instance는 “m5.large”를 선택합니다. 나머지는 기본값을 유지하고, 하단으로 스크롤하여 [Create]를 선택합니다.

![noname](https://github.com/kyopark2014/chatbot-based-on-Falcon-FM/assets/52392004/7c20d80c-52fc-4d18-b673-bd85e2660850)

2) [Environment](https://ap-northeast-2.console.aws.amazon.com/cloud9control/home?region=ap-northeast-2#/)에서 “chatbot”를 [Open]한 후에 아래와 같이 터미널을 실행합니다.

![noname](https://github.com/kyopark2014/chatbot-based-on-Falcon-FM/assets/52392004/b7d0c3c0-3e94-4126-b28d-d269d2635239)

3) EBS 크기 변경

아래와 같이 스크립트를 다운로드 합니다. 

```text
curl https://raw.githubusercontent.com/kyopark2014/technical-summary/main/resize.sh -o resize.sh
```

이후 아래 명령어로 용량을 80G로 변경합니다.
```text
chmod a+rx resize.sh && ./resize.sh 80
```


4) 소스를 다운로드합니다.

```java
git clone https://github.com/kyopark2014/rag-enhanced-searching
```

5) cdk 폴더로 이동하여 필요한 라이브러리를 설치합니다.

```java
cd rag-enhanced-searching/cdk-rag-enhanced-searching/ && npm install
```

6) CDK 사용을 위해 Boostraping을 수행합니다.

아래 명령어로 Account ID를 확인합니다.

```java
aws sts get-caller-identity --query Account --output text
```

아래와 같이 bootstrap을 수행합니다. 여기서 [Account-id]는 상기 명령어로 확인한 12자리의 Account ID입니다. bootstrap 1회만 수행하면 되므로, 기존에 cdk를 사용하고 있었다면 bootstrap은 건너뛰어도 됩니다.

```java
cdk bootstrap aws://[Account-id]/ap-northeast-1
```

7) 인프라를 설치합니다.

```java
cdk deploy --all
```

설치가 완료되면 아래와 같은 Output이 나옵니다. 

![noname](https://github.com/kyopark2014/rag-enhanced-searching/assets/52392004/bd032613-f546-48f0-8217-67e87d23789c)

8) HTMl 파일을 S3에 복사합니다.

아래와 같이 Output의 HtmlUpdateCommend을 붙여넣기 합니다. 

![noname](https://github.com/kyopark2014/rag-enhanced-searching/assets/52392004/307cb0a0-a09c-485c-a7a5-f65cef2c9eaf)


9) Google API Key Update하기

[api_key](https://developers.google.com/custom-search/docs/paid_element?hl=ko#api_key)에서 [키 가져오기] - [Select or create project]를 선택하여 Google API Key를 가져옵니다. 만약 기존 키가 없다면 새로 생성합니다.

[새 검색엔진 만들기](https://programmablesearchengine.google.com/controlpanel/create?hl=ko)에서 검색엔진을 설정합니다. 이때, 검색할 내용은 "전체 웹 검색"을 선택하여야 합니다.

[Secret Console](https://ap-northeast-1.console.aws.amazon.com/secretsmanager/secret?name=googl_api_key&region=ap-northeast-1)에 접속하여 [Retrieve secret value]를 선택하여, google_api_key와 google_cse_id를 업데이트합니다.

10) Output의 WebUrlforragenhancedsearching 있는 URL을 복사하여 웹 브라우저로 접속합니다. User Id로 적당한 이름을 넣고, Conversation Type로는 "2. Question/Answering (RAG)"를 선택합니다.

![noname](https://github.com/kyopark2014/rag-enhanced-searching/assets/52392004/689445d3-d4a1-4b2e-9390-303b0404c99b)
