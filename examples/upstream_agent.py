"""Example upstream agent for the LangGraph shim.

Deploy this service behind the shim:

- agent-chat-ui -> langgraph-shim -> this app
- this app only needs a simple `/chat` endpoint
- the shim handles LangGraph-compatible thread, history, and run APIs

Environment variables:

- ANTHROPIC_API_KEY: required by ChatAnthropic
- ANTHROPIC_MODEL: model name to use
- ANTHROPIC_MAX_TOKENS: max output tokens
- ANTHROPIC_TEMPERATURE: sampling temperature
- SYSTEM_PROMPT: system message prepended to each request
- HOST / PORT: bind address for uvicorn
"""

from __future__ import annotations

import json
import os
from typing import Any

import uvicorn
from fastapi import Body, FastAPI
from pydantic import BaseModel

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage


def env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    stripped = value.strip()
    return stripped if stripped else default


def env_int(name: str, default: int) -> int:
    value = env(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    value = env(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                for key in ("text", "content", "value"):
                    value = item.get(key)
                    if isinstance(value, str):
                        parts.append(value)
                        break
        if parts:
            return "".join(parts)
    if isinstance(content, dict):
        for key in ("text", "content", "message", "value"):
            value = content.get(key)
            if isinstance(value, str):
                return value
    return json.dumps(content, ensure_ascii=False, default=str)


def normalize_message(item: Any) -> BaseMessage | None:
    if not isinstance(item, dict):
        return None

    role = str(item.get("role") or item.get("type") or "").lower().strip()
    content = content_to_text(item.get("content")).strip()
    if not content:
        return None

    if role in {"user", "human"}:
        return HumanMessage(content=content)
    if role in {"assistant", "ai"}:
        return AIMessage(content=content)
    if role == "system":
        return SystemMessage(content=content)
    if role == "tool":
        tool_call_id = item.get("tool_call_id") or item.get("toolCallId")
        if tool_call_id:
            return ToolMessage(content=content, tool_call_id=str(tool_call_id))
    return None


def extract_prompt(payload: dict[str, Any]) -> str:
    for key in ("message", "prompt", "text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    input_values = payload.get("input")
    if isinstance(input_values, dict):
        for key in ("message", "prompt", "text"):
            value = input_values.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    for key in ("messages",):
        messages = payload.get(key)
        if isinstance(messages, list):
            for item in reversed(messages):
                msg = normalize_message(item)
                if isinstance(msg, HumanMessage):
                    return content_to_text(msg.content).strip()

    state_values = payload.get("state")
    if isinstance(state_values, dict):
        messages = state_values.get("messages")
        if isinstance(messages, list):
            for item in reversed(messages):
                msg = normalize_message(item)
                if isinstance(msg, HumanMessage):
                    return content_to_text(msg.content).strip()

    return ""


def build_messages(payload: dict[str, Any], system_prompt: str) -> list[BaseMessage]:
    raw_messages: list[Any] = []

    messages = payload.get("messages")
    if isinstance(messages, list):
        raw_messages = messages
    else:
        input_values = payload.get("input")
        if isinstance(input_values, dict) and isinstance(input_values.get("messages"), list):
            raw_messages = input_values["messages"]
        else:
            state_values = payload.get("state")
            if isinstance(state_values, dict) and isinstance(state_values.get("messages"), list):
                raw_messages = state_values["messages"]

    result: list[BaseMessage] = []
    for item in raw_messages:
        msg = normalize_message(item)
        if msg is not None:
            result.append(msg)

    if system_prompt and not any(isinstance(msg, SystemMessage) for msg in result):
        result.insert(0, SystemMessage(content=system_prompt))

    if not any(isinstance(msg, HumanMessage) for msg in result):
        prompt = extract_prompt(payload) or "Please respond to the user's message."
        result.append(HumanMessage(content=prompt))

    return result


def serialize_assistant_message(text: str) -> list[dict[str, Any]]:
    return [{"type": "assistant", "role": "assistant", "content": text}]


class ChatResponse(BaseModel):
    response: str
    messages: list[dict[str, Any]]
    model: str
    thread_id: str | None = None


class UpstreamAgent:
    def __init__(self) -> None:
        self.model_name = env("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001") or "claude-haiku-4-5-20251001"
        self.system_prompt = env("SYSTEM_PROMPT", "You are a helpful AI assistant.") or "You are a helpful AI assistant."
        self.max_tokens = env_int("ANTHROPIC_MAX_TOKENS", 1024)
        self.temperature = env_float("ANTHROPIC_TEMPERATURE", 0.0)
        self.llm = ChatAnthropic(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

    async def predict(self, payload: dict[str, Any]) -> ChatResponse:
        messages = build_messages(payload, self.system_prompt)
        result = await self.llm.ainvoke(messages)
        response_text = content_to_text(getattr(result, "content", "")).strip()
        if not response_text:
            response_text = "I do not have a response."

        return ChatResponse(
            response=response_text,
            messages=serialize_assistant_message(response_text),
            model=self.model_name,
            thread_id=payload.get("thread_id") if isinstance(payload.get("thread_id"), str) else None,
        )


agent = UpstreamAgent()
app = FastAPI(title="Upstream Agent", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    return {"status": "ready"}


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: dict[str, Any] | None = Body(default=None)) -> ChatResponse:
    return await agent.predict(payload or {})


def main() -> None:
    host = env("HOST", "0.0.0.0") or "0.0.0.0"
    port = env_int("PORT", 8080)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
