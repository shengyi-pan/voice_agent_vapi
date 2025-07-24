# Voice Agent VAPI Server

基于 LangChain 和 LangGraph 的语音代理服务器，提供 OpenAI 兼容的 /chat/completions 接口。

## 功能特性

- 使用 LangGraph 的 React Agent 实现智能对话
- 支持转人工功能（transferCall 工具）
- 提供 OpenAI 兼容的 /chat/completions 接口
- 支持流式和非流式响应
- API Token 认证
- 健康检查接口

## 项目结构

```
voice_agent_vapi/
├── main.py                  # 主入口文件
├── src/                     # 源代码目录
│   ├── __init__.py
│   ├── agent/               # Agent 实现
│   │   ├── __init__.py
│   │   └── agent.py         # React Agent 和工具定义
│   └── server/              # FastAPI 服务器
│       ├── __init__.py
│       └── app.py           # FastAPI 应用和端点
├── test/                    # 测试代码
│   ├── __init__.py
│   ├── test_server.py       # 服务器测试脚本
│   └── run_tests.py         # 测试运行器
├── .env                     # 环境变量配置
├── pyproject.toml           # 项目依赖
├── uv.lock                  # 依赖锁文件
└── README.md                # 项目说明文档
```

## 快速开始

### 1. 安装依赖

```bash
uv sync
```

### 2. 配置环境变量

编辑 `.env` 文件，设置必要的API密钥：

```env
OPENAI_API_KEY=your_openai_api_key_here
API_TOKEN=your_api_token_here
MODEL_NAME=gpt-3.5-turbo
TRANSFER_PHONE_NUMBER=+1234567890
```

### 3. 启动服务器

```bash
python main.py
```

服务器将在 `http://localhost:8000` 启动。

### 4. 测试服务

运行测试脚本：

```bash
cd test && python run_tests.py
```

## API 接口

### 1. 聊天完成接口

**POST** `/chat/completions`

OpenAI 兼容的聊天完成接口。

**请求头：**
```
Authorization: Bearer <your_api_token>
Content-Type: application/json
```

**请求体：**
```json
{
  "model": "gpt-3.5-turbo",
  "messages": [
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "stream": false
}
```

**响应（非流式）：**
```json
{
  "id": "chatcmpl-1234567890",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "gpt-3.5-turbo",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! I'm doing well, thank you for asking. How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ]
}
```

**流式响应：**
设置 `"stream": true` 将返回 Server-Sent Events 格式的流式响应。

### 2. 健康检查接口

**GET** `/health`

返回服务器健康状态。

**响应：**
```json
{
  "status": "healthy",
  "timestamp": 1234567890
}
```

## 转人工功能

当用户表达想要转人工时，Agent 会自动调用 `transferCall` 工具。该工具会返回转接信息，例如：

用户输入：
```
"I want to speak to a human agent"
```

Agent 响应：
```
"I have initiated the transfer to a human agent."
```

## 技术实现

### 架构设计

- **入口点**: `main.py` - 简单的服务器启动入口
- **Agent 模块**: `src/agent/agent.py` - 包含 React Agent 和工具定义
- **服务器模块**: `src/server/app.py` - FastAPI 应用和所有端点
- **测试模块**: `test/` - 完整的测试套件

### 技术栈

- **框架**: FastAPI + LangChain + LangGraph
- **Agent**: LangGraph React Agent
- **LLM**: OpenAI GPT-3.5-turbo
- **认证**: Bearer Token
- **流式响应**: Server-Sent Events (SSE)

## 测试

项目包含完整的测试套件：

- `test/test_server.py`: 基础功能测试
- `test/run_tests.py`: 启动服务器并运行测试

测试覆盖：
- 健康检查
- 非流式聊天
- 流式聊天
- 转人工功能

运行方式：
```bash
cd test && python run_tests.py  # 自动启动服务器并运行所有测试
```

## 开发指南

### 添加新工具

在 `src/agent/agent.py` 中添加新的工具函数：

```python
@tool
def your_new_tool(param: str) -> str:
    """Your tool description"""
    # 实现工具逻辑
    return result

# 然后在 create_voice_agent 函数中添加到 tools 列表
tools = [transferCall, your_new_tool]
```

### 添加新端点

在 `src/server/app.py` 中添加新的 FastAPI 端点：

```python
@app.get("/your-endpoint")
async def your_endpoint():
    return {"message": "Hello World"}
```

### 模块化导入

项目使用模块化结构，可以独立导入各个组件：

```python
from src.agent.agent import create_voice_agent
from src.server.app import app
```

## 注意事项

1. 确保设置正确的 OpenAI API Key
2. API Token 用于接口认证，请妥善保管
3. 转人工电话号码可在 .env 文件中配置
4. 服务器默认运行在 8000 端口
5. 所有测试需要在 test 目录下运行