# CDK로 인프라 설치하기

S3를 생성합니다.

```typescript
const s3Bucket = new s3.Bucket(this, `storage-${projectName}`, {
    bucketName: bucketName,
    blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
    removalPolicy: cdk.RemovalPolicy.DESTROY,
    autoDeleteObjects: true,
    publicReadAccess: false,
    versioned: false,
    cors: [
        {
            allowedHeaders: ['*'],
            allowedMethods: [
                s3.HttpMethods.POST,
                s3.HttpMethods.PUT,
            ],
            allowedOrigins: ['*'],
        },
    ],
});
```

CloudFront를 생성합니다.

```typescript
const distribution = new cloudFront.Distribution(this, `cloudfront-for-${projectName}`, {
    defaultBehavior: {
        origin: new origins.S3Origin(s3Bucket),
        allowedMethods: cloudFront.AllowedMethods.ALLOW_ALL,
        cachePolicy: cloudFront.CachePolicy.CACHING_DISABLED,
        viewerProtocolPolicy: cloudFront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
    },
    priceClass: cloudFront.PriceClass.PRICE_CLASS_200,
});
```

DynamoDB를 생성합니다.

```typescript
const callLogTableName = `db-call-log-for-${projectName}`;
const callLogDataTable = new dynamodb.Table(this, `db-call-log-for-${projectName}`, {
    tableName: callLogTableName,
    partitionKey: { name: 'user_id', type: dynamodb.AttributeType.STRING },
    sortKey: { name: 'request_time', type: dynamodb.AttributeType.STRING },
    billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
    removalPolicy: cdk.RemovalPolicy.DESTROY,
});
const callLogIndexName = `index-type-for-${projectName}`;
callLogDataTable.addGlobalSecondaryIndex({ // GSI
    indexName: callLogIndexName,
    partitionKey: { name: 'request_id', type: dynamodb.AttributeType.STRING },
});
```

Lambda(chat)을 위한 Role을 생성합니다.

```typescript
const roleLambdaWebsocket = new iam.Role(this, `role-lambda-chat-ws-for-${projectName}`, {
    roleName: `role-lambda-chat-ws-for-${projectName}-${region}`,
    assumedBy: new iam.CompositePrincipal(
        new iam.ServicePrincipal("lambda.amazonaws.com"),
        new iam.ServicePrincipal("bedrock.amazonaws.com"),
    )
});
roleLambdaWebsocket.addManagedPolicy({
    managedPolicyArn: 'arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
});
const BedrockPolicy = new iam.PolicyStatement({  // policy statement for sagemaker
    resources: ['*'],
    actions: ['bedrock:*'],
});
roleLambdaWebsocket.attachInlinePolicy( // add bedrock policy
    new iam.Policy(this, `bedrock-policy-lambda-chat-ws-for-${projectName}`, {
        statements: [BedrockPolicy],
    }),
);
const apiInvokePolicy = new iam.PolicyStatement({
    // resources: ['arn:aws:execute-api:*:*:*'],
    resources: ['*'],
    actions: [
        'execute-api:Invoke',
        'execute-api:ManageConnections'
    ],
});
roleLambdaWebsocket.attachInlinePolicy(
    new iam.Policy(this, `api-invoke-policy-for-${projectName}`, {
        statements: [apiInvokePolicy],
    }),
);
```

OpenSearch를 생성합니다.

```typescript
const domainName = projectName
const accountId = process.env.CDK_DEFAULT_ACCOUNT;
const resourceArn = `arn:aws:es:${region}:${accountId}:domain/${domainName}/*`

const OpenSearchAccessPolicy = new iam.PolicyStatement({
    resources: [resourceArn],
    actions: ['es:*'],
    effect: iam.Effect.ALLOW,
    principals: [new iam.AnyPrincipal()],
});

const domain = new opensearch.Domain(this, 'Domain', {
    version: opensearch.EngineVersion.OPENSEARCH_2_3,

    domainName: domainName,
    removalPolicy: cdk.RemovalPolicy.DESTROY,
    enforceHttps: true,
    fineGrainedAccessControl: {
        masterUserName: opensearch_account,
        // masterUserPassword: cdk.SecretValue.secretsManager('opensearch-private-key'),
        masterUserPassword: cdk.SecretValue.unsafePlainText(opensearch_passwd)
    },
    capacity: {
        masterNodes: 3,
        masterNodeInstanceType: 'm6g.large.search',
        // multiAzWithStandbyEnabled: false,
        dataNodes: 3,
        dataNodeInstanceType: 'r6g.large.search',
        // warmNodes: 2,
        // warmInstanceType: 'ultrawarm1.medium.search',
    },
    accessPolicies: [OpenSearchAccessPolicy],
    ebs: {
        volumeSize: 100,
        volumeType: ec2.EbsDeviceVolumeType.GP3,
    },
    nodeToNodeEncryption: true,
    encryptionAtRest: {
        enabled: true,
    },
    zoneAwareness: {
        enabled: true,
        availabilityZoneCount: 3,
    }
});
```

API Gateway를 위한 Role을 생성합니다.

```typescript
const role = new iam.Role(this, `api-role-for-${projectName}`, {
    roleName: `api-role-for-${projectName}-${region}`,
    assumedBy: new iam.ServicePrincipal("apigateway.amazonaws.com")
});
role.addToPolicy(new iam.PolicyStatement({
    resources: ['*'],
    actions: [
        'lambda:InvokeFunction',
        'cloudwatch:*'
    ]
}));
role.addManagedPolicy({
    managedPolicyArn: 'arn:aws:iam::aws:policy/AWSLambdaExecute',
});

const api = new apiGateway.RestApi(this, `api-chatbot-for-${projectName}`, {
    description: 'API Gateway for chatbot',
    endpointTypes: [apiGateway.EndpointType.REGIONAL],
    binaryMediaTypes: ['application/pdf', 'text/plain', 'text/csv', 'application/vnd.ms-powerpoint', 'application/vnd.ms-excel', 'application/msword'],
    deployOptions: {
        stageName: stage,
    },
});
```

파일 업로드를 위한 Lambda(upload)를 생성합니다.

```typescript
const lambdaUpload = new lambda.Function(this, `lambda-upload-for-${projectName}`, {
    runtime: lambda.Runtime.NODEJS_16_X,
    functionName: `lambda-upload-for-${projectName}`,
    code: lambda.Code.fromAsset("../lambda-upload"),
    handler: "index.handler",
    timeout: cdk.Duration.seconds(10),
    environment: {
        bucketName: s3Bucket.bucketName,
        s3_prefix: s3_prefix
    }
});
s3Bucket.grantReadWrite(lambdaUpload);

const resourceName = "upload";
const upload = api.root.addResource(resourceName);
upload.addMethod('POST', new apiGateway.LambdaIntegration(lambdaUpload, {
    passthroughBehavior: apiGateway.PassthroughBehavior.WHEN_NO_TEMPLATES,
    credentialsRole: role,
    integrationResponses: [{
        statusCode: '200',
    }],
    proxy: false,
}), {
    methodResponses: [
        {
            statusCode: '200',
            responseModels: {
                'application/json': apiGateway.Model.EMPTY_MODEL,
            },
        }
    ]
});

distribution.addBehavior("/upload", new origins.RestApiOrigin(api), {
    cachePolicy: cloudFront.CachePolicy.CACHING_DISABLED,
    allowedMethods: cloudFront.AllowedMethods.ALLOW_ALL,
    viewerProtocolPolicy: cloudFront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
});
```

Client에서 DynamoDB을 조회하기 위한 Lambda를 정의합니다. 

```typescript
const lambdaQueryResult = new lambda.Function(this, `lambda-query-for-${projectName}`, {
    runtime: lambda.Runtime.NODEJS_16_X,
    functionName: `lambda-query-for-${projectName}`,
    code: lambda.Code.fromAsset("../lambda-query"),
    handler: "index.handler",
    timeout: cdk.Duration.seconds(60),
    environment: {
        tableName: callLogTableName,
        indexName: callLogIndexName
    }
});
callLogDataTable.grantReadWriteData(lambdaQueryResult); // permission for dynamo

// POST method - query
const query = api.root.addResource("query");
query.addMethod('POST', new apiGateway.LambdaIntegration(lambdaQueryResult, {
    passthroughBehavior: apiGateway.PassthroughBehavior.WHEN_NO_TEMPLATES,
    credentialsRole: role,
    integrationResponses: [{
        statusCode: '200',
    }],
    proxy: false,
}), {
    methodResponses: [
        {
            statusCode: '200',
            responseModels: {
                'application/json': apiGateway.Model.EMPTY_MODEL,
            },
        }
    ]
});

// cloudfront setting for api gateway    
distribution.addBehavior("/query", new origins.RestApiOrigin(api), {
    cachePolicy: cloudFront.CachePolicy.CACHING_DISABLED,
    allowedMethods: cloudFront.AllowedMethods.ALLOW_ALL,
    viewerProtocolPolicy: cloudFront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
});
```


채팅 이력을 조회하기 위한 Lambda(history)를 정의합니다.

```typescript
const lambdaGetHistory = new lambda.Function(this, `lambda-gethistory-for-${projectName}`, {
    runtime: lambda.Runtime.NODEJS_16_X,
    functionName: `lambda-gethistory-for-${projectName}`,
    code: lambda.Code.fromAsset("../lambda-gethistory"),
    handler: "index.handler",
    timeout: cdk.Duration.seconds(60),
    environment: {
        tableName: callLogTableName
    }
});
callLogDataTable.grantReadWriteData(lambdaGetHistory); // permission for dynamo

// POST method - history
const history = api.root.addResource("history");
history.addMethod('POST', new apiGateway.LambdaIntegration(lambdaGetHistory, {
    passthroughBehavior: apiGateway.PassthroughBehavior.WHEN_NO_TEMPLATES,
    credentialsRole: role,
    integrationResponses: [{
        statusCode: '200',
    }],
    proxy: false,
}), {
    methodResponses: [
        {
            statusCode: '200',
            responseModels: {
                'application/json': apiGateway.Model.EMPTY_MODEL,
            },
        }
    ]
});

// cloudfront setting for api gateway    
distribution.addBehavior("/history", new origins.RestApiOrigin(api), {
    cachePolicy: cloudFront.CachePolicy.CACHING_DISABLED,
    allowedMethods: cloudFront.AllowedMethods.ALLOW_ALL,
    viewerProtocolPolicy: cloudFront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
});
```

메시지를 삭제하기 위한 Lambda(delete)를 정의합니다. 

```typescript
const lambdaDeleteItems = new lambda.Function(this, `lambda-deleteItems-for-${projectName}`, {
    runtime: lambda.Runtime.NODEJS_16_X,
    functionName: `lambda-deleteItems-for-${projectName}`,
    code: lambda.Code.fromAsset("../lambda-delete-items"),
    handler: "index.handler",
    timeout: cdk.Duration.seconds(60),
    environment: {
        tableName: callLogTableName
    }
});
callLogDataTable.grantReadWriteData(lambdaDeleteItems); // permission for dynamo

// POST method - delete items
const deleteItem = api.root.addResource("delete");
deleteItem.addMethod('POST', new apiGateway.LambdaIntegration(lambdaDeleteItems, {
    passthroughBehavior: apiGateway.PassthroughBehavior.WHEN_NO_TEMPLATES,
    credentialsRole: role,
    integrationResponses: [{
        statusCode: '200',
    }],
    proxy: false,
}), {
    methodResponses: [
        {
            statusCode: '200',
            responseModels: {
                'application/json': apiGateway.Model.EMPTY_MODEL,
            },
        }
    ]
});

// cloudfront setting for api gateway    
distribution.addBehavior("/delete", new origins.RestApiOrigin(api), {
    cachePolicy: cloudFront.CachePolicy.CACHING_DISABLED,
    allowedMethods: cloudFront.AllowedMethods.ALLOW_ALL,
    viewerProtocolPolicy: cloudFront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
});
```

WebSocket을 위한 API Gateway를 생성합니다.


```typescript
const websocketapi = new apigatewayv2.CfnApi(this, `ws-api-for-${projectName}`, {
    description: 'API Gateway for chatbot using websocket',
    apiKeySelectionExpression: "$request.header.x-api-key",
    name: 'api-' + projectName,
    protocolType: "WEBSOCKET", // WEBSOCKET or HTTP
    routeSelectionExpression: "$request.body.action",
});
websocketapi.applyRemovalPolicy(cdk.RemovalPolicy.DESTROY); // DESTROY, RETAIN

const wss_url = `wss://${websocketapi.attrApiId}.execute-api.${region}.amazonaws.com/${stage}`;
new cdk.CfnOutput(this, 'web-socket-url', {
    value: wss_url,
    description: 'The URL of Web Socket',
});

const connection_url = `https://${websocketapi.attrApiId}.execute-api.${region}.amazonaws.com/${stage}`;
if (debug) {
    new cdk.CfnOutput(this, 'api-identifier', {
        value: websocketapi.attrApiId,
        description: 'The API identifier.',
    });

    new cdk.CfnOutput(this, 'connection-url', {
        value: connection_url,
        description: 'The URL of connection',
    });
}
```

Lambda(chat)을 정의합니다.

```typescript
const lambdaChatWebsocket = new lambda.DockerImageFunction(this, `lambda-chat-ws-for-${projectName}`, {
    description: 'lambda for chat using websocket',
    functionName: `lambda-chat-ws-for-${projectName}`,
    code: lambda.DockerImageCode.fromImageAsset(path.join(__dirname, '../../lambda-chat-ws')),
    timeout: cdk.Duration.seconds(300),
    memorySize: 8192,
    role: roleLambdaWebsocket,
    environment: {
        s3_bucket: s3Bucket.bucketName,
        s3_prefix: s3_prefix,
        callLogTableName: callLogTableName,
        connection_url: connection_url,
        enableReference: enableReference,
        opensearch_account: opensearch_account,
        opensearch_passwd: opensearch_passwd,
        opensearch_url: opensearch_url,
        path: 'https://' + distribution.domainName + '/',
        roleArn: roleLambdaWebsocket.roleArn,
        numberOfRelevantDocs: numberOfRelevantDocs,
        profile_of_LLMs: profile_of_LLMs,
        allowDualSearching: allowDualSearching,
        enableNoriPlugin: enableNoriPlugin,
        enableParallelSummay: enableParallelSummay
    }
});
lambdaChatWebsocket.grantInvoke(new iam.ServicePrincipal('apigateway.amazonaws.com'));
s3Bucket.grantReadWrite(lambdaChatWebsocket); // permission for s3
callLogDataTable.grantReadWriteData(lambdaChatWebsocket); // permission for dynamo 
```

API Gateway를 설정합니다.

```typescript
const integrationUri = `arn:aws:apigateway:${region}:lambda:path/2015-03-31/functions/${lambdaChatWebsocket.functionArn}/invocations`;
const cfnIntegration = new apigatewayv2.CfnIntegration(this, `api-integration-for-${projectName}`, {
    apiId: websocketapi.attrApiId,
    integrationType: 'AWS_PROXY',
    credentialsArn: role.roleArn,
    connectionType: 'INTERNET',
    description: 'Integration for connect',
    integrationUri: integrationUri,
});

new apigatewayv2.CfnRoute(this, `api-route-for-${projectName}-connect`, {
    apiId: websocketapi.attrApiId,
    routeKey: "$connect",
    apiKeyRequired: false,
    authorizationType: "NONE",
    operationName: 'connect',
    target: `integrations/${cfnIntegration.ref}`,
});

new apigatewayv2.CfnRoute(this, `api-route-for-${projectName}-disconnect`, {
    apiId: websocketapi.attrApiId,
    routeKey: "$disconnect",
    apiKeyRequired: false,
    authorizationType: "NONE",
    operationName: 'disconnect',
    target: `integrations/${cfnIntegration.ref}`,
});

new apigatewayv2.CfnRoute(this, `api-route-for-${projectName}-default`, {
    apiId: websocketapi.attrApiId,
    routeKey: "$default",
    apiKeyRequired: false,
    authorizationType: "NONE",
    operationName: 'default',
    target: `integrations/${cfnIntegration.ref}`,
});

new apigatewayv2.CfnStage(this, `api-stage-for-${projectName}`, {
    apiId: websocketapi.attrApiId,
    stageName: stage
});
```

S3의 파일 삭제 이벤트를 조회하기 위한 Lambda를 정의합니다.

```typescript
const lambdaS3event = new lambda.DockerImageFunction(this, `lambda-S3-event-for-${projectName}`, {
    description: 'S3 event',
    functionName: `lambda-s3-event-for-${projectName}`,
    code: lambda.DockerImageCode.fromImageAsset(path.join(__dirname, '../../lambda-s3-event')),
    timeout: cdk.Duration.seconds(60),
    environment: {
        s3_bucket: s3Bucket.bucketName,
        s3_prefix: s3_prefix,
        opensearch_account: opensearch_account,
        opensearch_passwd: opensearch_passwd,
        opensearch_url: opensearch_url,
        roleArn: roleLambdaWebsocket.roleArn,
    }
});
s3Bucket.grantReadWrite(lambdaS3event); // permission for s3

// s3 event source
const s3PutEventSource = new lambdaEventSources.S3EventSource(s3Bucket, {
    events: [
        // s3.EventType.OBJECT_CREATED_PUT,
        s3.EventType.OBJECT_REMOVED_DELETE
    ],
    filters: [
        { prefix: s3_prefix + '/' },
    ]
});
lambdaS3event.addEventSource(s3PutEventSource);
```

Provisioning을 위한 Lambda를 정의합니다.

```typescript
const lambdaProvisioning = new lambda.Function(this, `lambda-provisioning-for-${projectName}`, {
    description: 'lambda to earn provisioning info',
    functionName: `lambda-provisioning-api-${projectName}`,
    handler: 'lambda_function.lambda_handler',
    runtime: lambda.Runtime.PYTHON_3_11,
    code: lambda.Code.fromAsset(path.join(__dirname, '../../lambda-provisioning')),
    timeout: cdk.Duration.seconds(30),
    environment: {
        wss_url: wss_url,
    }
});

// POST method - provisioning
const provisioning_info = api.root.addResource("provisioning");
provisioning_info.addMethod('POST', new apiGateway.LambdaIntegration(lambdaProvisioning, {
    passthroughBehavior: apiGateway.PassthroughBehavior.WHEN_NO_TEMPLATES,
    credentialsRole: role,
    integrationResponses: [{
        statusCode: '200',
    }],
    proxy: false,
}), {
    methodResponses: [
        {
            statusCode: '200',
            responseModels: {
                'application/json': apiGateway.Model.EMPTY_MODEL,
            },
        }
    ]
});

distribution.addBehavior("/provisioning", new origins.RestApiOrigin(api), {
    cachePolicy: cloudFront.CachePolicy.CACHING_DISABLED,
    allowedMethods: cloudFront.AllowedMethods.ALLOW_ALL,
    viewerProtocolPolicy: cloudFront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
});
```

API Gateway를 배포합니다. 

```typescript
new componentDeployment(scope, `component-deployment-of-${projectName}`, websocketapi.attrApiId)
export class componentDeployment extends cdk.Stack {
    constructor(scope: Construct, id: string, appId: string, props?: cdk.StackProps) {
        super(scope, id, props);

        new apigatewayv2.CfnDeployment(this, `api-deployment-of-${projectName}`, {
            apiId: appId,
            description: "deploy api gateway using websocker",  // $default
            stageName: stage
        });
    }
} 
```
