"""Example agent script that uses the shim directly.

Install the `langgraph-shim` package into the `python-agent-pipeline` image,
then deploy this script as the agent entrypoint.

The deployed pod exposes the LangGraph-compatible shim routes and calls the
local `predict()` method whenever the UI sends a chat request.
"""

from __future__ import annotations

import os
from typing import TypedDict

import uvicorn
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from langgraph_shim.app import app, set_upstream_handler


class GraphState(TypedDict):
    question: str
    answer: str


class LangGraphPredictor:
    def __init__(self) -> None:
        self.llm = ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
            max_tokens=int(os.getenv("ANTHROPIC_MAX_TOKENS", "1024")),
            temperature=float(os.getenv("ANTHROPIC_TEMPERATURE", "0.0")),
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("call_llm", self.call_llm)
        graph.add_edge(START, "call_llm")
        graph.add_edge("call_llm", END)
        return graph.compile()

    async def call_llm(self, state: GraphState):
        messages = [
            SystemMessage(content="You are a helpful AI assistant."),
            HumanMessage(content=state["question"]),
        ]
        result = await self.llm.ainvoke(messages)
        return {"answer": result.content}

    async def predict(self, payload: dict):
        message = payload.get("message", "")
        result = await self.graph.ainvoke({"question": message, "answer": ""})
        return {
            "response": result["answer"],
            "messages": [{"type": "assistant", "content": result["answer"]}],
        }


predictor = LangGraphPredictor()
set_upstream_handler(predictor.predict)


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))


if __name__ == "__main__":
    main()
