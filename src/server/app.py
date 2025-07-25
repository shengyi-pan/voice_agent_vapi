"""
FastAPI server implementation for Voice Agent VAPI
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk

from ..agent.agent import create_voice_agent, is_transfer_tool


# Load environment variables
load_dotenv(find_dotenv())


# Pydantic models for request/response
class ToolCall(BaseModel):
    id: str
    type: str
    function: Dict[str, Any]


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    name: Optional[str] = None  # Used for function/tool responses
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None  # Used for tool responses


class Function(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class Tool(BaseModel):
    type: str = "function"
    function: Function


class ResponseFormat(BaseModel):
    type: str = "text"  # "text", "json_object", "json_schema"
    json_schema: Optional[Dict[str, Any]] = None


class ChatCompletionRequest(BaseModel):
    # Required parameters
    model: str
    messages: List[ChatMessage]

    # Optional parameters (matching OpenAI API)
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    n: Optional[int] = 1
    presence_penalty: Optional[float] = 0.0
    response_format: Optional[ResponseFormat] = None
    seed: Optional[int] = None
    service_tier: Optional[str] = None
    stop: Optional[List[str]] = None
    stream: Optional[bool] = False
    stream_options: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[str] = None  # "none", "auto", "required", or specific tool
    parallel_tool_calls: Optional[bool] = True
    user: Optional[str] = None

    # Additional fields for compatibility
    function_call: Optional[str] = None  # Deprecated but still supported
    functions: Optional[List[Function]] = None  # Deprecated but still supported


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    index: int
    message: Optional[ChatMessage] = None
    delta: Optional[Dict[str, Any]] = None
    logprobs: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    system_fingerprint: Optional[str] = None
    choices: List[ChatCompletionChoice]
    usage: Optional[Usage] = None


# Initialize FastAPI app
app = FastAPI(title="Voice Agent VAPI", version="1.0.0")

# Global agent instance
agent = create_voice_agent()


# Authentication
def verify_token(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")

    token = authorization.replace("Bearer ", "")
    expected_token = os.getenv("API_TOKEN", "your_api_token_here")

    if token != expected_token:
        raise HTTPException(status_code=401, detail="Invalid token")

    return token


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest, _: str = Depends(verify_token)
):
    """OpenAI-compatible chat completions endpoint"""

    # Convert messages to LangGraph format
    messages = []
    for msg in request.messages:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content or ""))
        elif msg.role == "assistant":
            messages.append(AIMessage(content=msg.content or ""))

    # Create thread config
    thread_config = {"configurable": {"thread_id": "default"}}

    if request.stream:
        return handle_streaming_req(messages, thread_config, request)

    else:
        return handle_non_streaming_req(messages, thread_config, request)


def handle_streaming_req(
    messages: List, thread_config: Dict[str, Any], request: ChatCompletionRequest
):
    """
    处理流式请求

    Args:
        messages: 消息列表
        thread_config: langchain state配置
        request: 请求对象
    """

    # Streaming response
    async def stream_response():
        try:
            # Generate unique response ID
            response_id = f"chatcmpl-{int(time.time())}"
            created_time = int(time.time())

            # First, send the initial chunk with role
            initial_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(initial_chunk)}\n\n"

            # Stream the LLM responses
            accumulated_chunk = None
            has_tool_calls = False
            tool_calls_sent = False
            ai_content_sent = False

            for event in agent.stream(
                {"messages": messages}, config=thread_config, stream_mode="messages"
            ):
                chunk = event[0]
                # 如果 Agent 自动调用工具， AIMessage会交替出现 tool调用 和 内容生成
                if isinstance(chunk, AIMessageChunk):
                    # 如果 content 有数据，就先返回
                    if chunk.content:
                        chunk_data = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": chunk.content},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"
                        ai_content_sent = True

                    # 如果有工具调用就触发chunk累积，可能会累积到最后的AIMessage
                    if chunk.tool_call_chunks or has_tool_calls:
                        has_tool_calls = True
                        # Accumulate chunks when tool calls are present
                        if accumulated_chunk is None:
                            accumulated_chunk = chunk
                        else:
                            accumulated_chunk += chunk

            is_contain_transfer = False
            # Handle accumulated tool calls
            if has_tool_calls and accumulated_chunk and not tool_calls_sent:
                # 如果累积的 AIMessage 有内容，且没有发送过 AI 内容，就先发送
                if accumulated_chunk.content and not ai_content_sent:
                    chunk_data = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": accumulated_chunk.content},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"

                # Then, send tool calls if any
                if accumulated_chunk.tool_call_chunks:
                    # Send tool calls in OpenAI SSE format
                    tool_calls_data = []
                    for i, tool_chunk in enumerate(accumulated_chunk.tool_call_chunks):
                        tool_call = {
                            "index": i,
                            "id": tool_chunk.get("id", f"call_{int(time.time())}_{i}"),
                            "type": "function",
                            "function": {
                                "name": tool_chunk.get("name", ""),
                                "arguments": json.dumps(tool_chunk.get("args", {})),
                            },
                        }

                        if is_transfer_tool(tool_chunk.get("name", "")):
                            is_contain_transfer = True

                        tool_calls_data.append(tool_call)

                    # Send tool calls chunk
                    tool_chunk_data = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"tool_calls": tool_calls_data},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(tool_chunk_data)}\n\n"
                    tool_calls_sent = True

            # Send finish chunk with proper finish reason
            # 如果工具调用包含转人工，finish_reason 为 tool_calls，否则为 stop
            if is_contain_transfer:
                finish_reason = "tool_calls"
            else:
                finish_reason = "stop"

            # Send finish chunk
            finish_tool_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason,
                    }
                ],
            }
            yield f"data: {json.dumps(finish_tool_chunk)}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            error_chunk = {"error": {"message": str(e), "type": "internal_error"}}
            yield f"data: {json.dumps(error_chunk)}\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Methods": "*",
        },
    )


def handle_non_streaming_req(
    messages: List, thread_config: Dict[str, Any], request: ChatCompletionRequest
):
    """
    处理非流式请求

    Args:
        messages: 消息列表
        thread_config: langchain state 配置
        request: 请求对象
    """

    try:
        result = agent.invoke({"messages": messages}, config=thread_config)

        # Extract the final message
        final_message = result["messages"][-1]

        # Generate response ID and timestamp
        response_id = f"chatcmpl-{int(time.time())}"
        created_time = int(time.time())

        # Determine finish reason and prepare message
        finish_reason = "stop"
        message_content = final_message.content
        tool_calls = None

        # Check if the message has tool calls
        if hasattr(final_message, "tool_calls") and final_message.tool_calls:
            finish_reason = "tool_calls"

            # Convert tool calls to OpenAI format
            tool_calls = []
            for tool_call in final_message.tool_calls:
                openai_tool_call = ToolCall(
                    id=tool_call.get(
                        "id", f"call_{int(time.time())}_{len(tool_calls)}"
                    ),
                    type="function",
                    function={
                        "name": tool_call.get("name", ""),
                        "arguments": json.dumps(tool_call.get("args", {})),
                    },
                )
                tool_calls.append(openai_tool_call)

        # Create the message object
        message = ChatMessage(
            role="assistant", content=message_content, tool_calls=tool_calls
        )

        # Calculate token usage (simplified estimation)
        # In a real implementation, you'd want to use tiktoken or similar
        prompt_text = " ".join([msg.content or "" for msg in request.messages])
        completion_text = message_content or ""

        # Rough token estimation (1 token ≈ 4 characters)
        prompt_tokens = max(1, len(prompt_text) // 4)
        completion_tokens = max(1, len(completion_text) // 4)
        total_tokens = prompt_tokens + completion_tokens

        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        response = ChatCompletionResponse(
            id=response_id,
            object="chat.completion",
            created=created_time,
            model=request.model,
            system_fingerprint=f"fp_{hash(request.model) % 1000000:06d}",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=message,
                    logprobs=None,
                    finish_reason=finish_reason,
                )
            ],
            usage=usage,
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": int(time.time())}


def run_server():
    """Run the FastAPI server"""
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
