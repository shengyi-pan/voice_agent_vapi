## 项目背景

使用 langchain，langgraph 实现 react agent，在此之上使用 FAST API 封装服务，提供 http 接口。
对外提供的接口必须和 OpenAI 的 /chat/completions 接口一致。接口会被上游的语音服务调用，视为 text-to-text agent。


## 实现目标

### React Agent

使用 `from langgraph.prebuilt import create_react_agent` 创建 react agent。这个 agent 可以简单定义为日常聊天 agent。

agent 需要配置 转人工 的工具，当用户表达想找人工时，agent 会调用这个工具。 工具的定义如下：

```json
{
  "function_call": {
    "name": "transferCall",
    "arguments": { "destination": "+1234567890" }
  }
}
```
destination 为电话号码。可以先固定为 +1234567890。


### 服务封装

使用 fastapi 封装服务，提供 http 接口。对外提供的必须和 OpenAI 的 /chat/completions 接口一致。

需要支持 streaming 模式， 参考的接口实现如下：

```python

import os
import json
import time  # Used for simulating a delay in streaming
from flask import Blueprint, request, Response, jsonify
from openai import OpenAI

custom_llm = Blueprint('custom_llm', __name__)

client = OpenAI(
  # This is the default and can be omitted
  api_key=os.environ.get("OPENAI_API_KEY"),
)



def generate_streaming_response(data):
  """
  Generator function to simulate streaming data.
  """
  for message in data:
    json_data = message.model_dump_json()
    yield f"data: {json_data}\n\n"




@custom_llm.route('/basic/chat/completions', methods=['POST'])
def basic_custom_llm_route():
  request_data = request.get_json()
  response = {
    "id": "chatcmpl-8mcLf78g0quztp4BMtwd3hEj58Uof",
    "object": "chat.completion",
    "created": int(time.time()),
    "model": "gpt-3.5-turbo-0613",
    "system_fingerprint": None,
    "choices": [
      {
        "index": 0,
        "delta": {"content": request_data['messages'][-1]['content'] if len(request_data['messages']) > 0 else ""},
        "logprobs": None,
        "finish_reason": "stop"
      }
    ]
  }
  return jsonify(response), 200




@custom_llm.route('/openai-sse/chat/completions', methods=['POST'])
def custom_llm_openai_sse_handler():
  request_data = request.get_json()
  streaming = request_data.get('stream', False)

  if streaming:
    # Simulate a stream of responses

    chat_completion_stream = client.chat.completions.create(**request_data)


    return Response(generate_streaming_response(chat_completion_stream), content_type='text/event-stream')
  else:
    # Simulate a non-streaming response
    chat_completion = client.chat.completions.create(**request_data)
    return Response(chat_completion.model_dump_json(), content_type='application/json')


@custom_llm.route('/openai-advanced/chat/completions', methods=['POST'])
def openai_advanced_custom_llm_route():
  request_data = request.get_json()
  streaming = request_data.get('stream', False)


  last_message = request_data['messages'][-1]
  prompt = f"""
    Create a prompt which can act as a prompt template where I put the original prompt and it can modify it according to my intentions so that the final modified prompt is more detailed. You can expand certain terms or keywords.
    ----------
    PROMPT: {last_message['content']}.
    MODIFIED PROMPT: """
  completion = client.completions.create(
    model= "gpt-3.5-turbo-instruct",
    prompt=prompt,
    max_tokens=500,
    temperature=0.7
  )
  modified_message = request_data['messages'][:-1] + [{'content': completion.choices[0].text, 'role': last_message['role']}]

  request_data['messages'] = modified_message
  if streaming:
    chat_completion_stream = client.chat.completions.create(**request_data)


    return Response(generate_streaming_response(chat_completion_stream), content_type='text/event-stream')
  else:
    # Simulate a non-streaming response
    chat_completion = client.chat.completions.create(**request_data)
    return Response(chat_completion.model_dump_json(), content_type='application/json')

```

#### 细节调整

1. 需要使用 Fast API 来封装服务

2. 使用 langgraph 实现和llm的交互， 不实用 openai python sdk
因为我底层的llm可能是来自多个供应商

3. streaming 效果使用 langgraph 的 streaming 功能

langgraph 的streaming 代码示例：

```python

user_input = {
    "messages": [
        {"role": "user", "content": "i wanna go somewhere warm in the caribbean"}
    ]
}

# Stream the LLM responses token by token
for event in graph.stream(
    user_input,
    config=thread_config,
    stream_mode="messages",  # Use "messages" for token-level streaming
):
    # print(len(event), type(event[0]), type(event[1]), event)
    chunk = event[0] # AIMessageChunk
    graph_meta = event[1] # GraphMetadata
    if isinstance(chunk, AIMessageChunk) and not chunk.tool_calls:
        
        if not chunk.tool_calls:
            # 应该以 SSE 的格式返回AI Content
            print(chunk.content, end="", flush=True)
            if chunk.response_metadata.get("finish_reason", "") == "stop":
                print("\n====END====\n")
        else：
            # 工具调用，例如 transferCall
            # 不用触发 SSE 返回，等生成结束返回完整内容
            

```

**注意要保持 SSE 格式的一致性**。


## 注意事项

1. 封装的接口都需要有 API Token 的验证，先简单固定为静态 token

2. 如果有不确定API Document， 可以使用 tavily 工具检索互联网， 使用 context7 检索github

3. 依赖服务 API Token 都统一存放在 .env 文件中
