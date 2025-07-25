"""
Voice Agent implementation using LangGraph React Agent
"""

import os
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field


class transferCallInput(BaseModel):
    destination: str = Field(
        description="The destination phone number. The default is +1234567890."
    )


@tool("transferCall", args_schema=transferCallInput)
def transferCall(destination: str = "+1234567890") -> str:
    """Transfer the call to a human agent."""
    return f"Transferring call to {destination}"


DEF_SYS_PROMPT = """
You are a helpful voice assistant, try to answer any user's question.

When user want to talk to human , call `transferCall` tool.

"""


def create_voice_agent(system_prompt: str = DEF_SYS_PROMPT):
    """Create and configure the React Agent with tools"""
    llm = init_chat_model(
        model=os.getenv("MODEL_NAME", "gpt-4.1"), model_provider="openai"
    )

    tools = [transferCall]
    agent = create_react_agent(llm, tools, prompt=system_prompt, interrupt_before=["tools"])
    # agent = create_react_agent(llm, tools, prompt=system_prompt)
    return agent


def get_transfer_phone_number():
    """Get the configured transfer phone number"""
    return os.getenv("TRANSFER_PHONE_NUMBER", "+1234567890")


def is_transfer_tool(tool_name: str) -> bool:
    """
    判断是否是转人工工具

    Args:
        tool_name: 工具名称
    """
    return tool_name == "transferCall"
