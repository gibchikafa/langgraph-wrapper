from __future__ import annotations

import asyncio
import json
import os
import uuid
from copy import deepcopy
from typing import Any, Callable

import httpx
from fastapi import Body, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from store import (
    LangGraphShimSettings,
    LangGraphShimStore,
    build_assistant_message,
    content_to_text,
    extract_text,
    find_last_human_message,
    json_dumps,
    json_loads,
    merge_values,
    now_iso,
)


def env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    stripped = value.strip()
    return stripped if stripped else default


def load_settings() -> LangGraphShimSettings:
    extra_headers_raw = env("UPSTREAM_EXTRA_HEADERS_JSON", "{}") or "{}"
    try:
        extra_headers_obj = json.loads(extra_headers_raw)
        if not isinstance(extra_headers_obj, dict):
            extra_headers_obj = {}
    except Exception:
        extra_headers_obj = {}

    upstream_api_key = (
        env("UPSTREAM_API_KEY")
        or env("LANGSMITH_API_KEY")
        or env("LANGGRAPH_API_KEY")
        or env("LANGCHAIN_API_KEY")
    )

    return LangGraphShimSettings(
        db_path=env("LANGGRAPH_SHIM_DB_PATH", "/tmp/langgraph-shim.sqlite3") or "/tmp/langgraph-shim.sqlite3",
        upstream_url=env("UPSTREAM_CHAT_URL", "http://127.0.0.1:8000/chat") or "http://127.0.0.1:8000/chat",
        upstream_method=(env("UPSTREAM_CHAT_METHOD", "POST") or "POST").upper(),
        upstream_timeout=float(env("UPSTREAM_TIMEOUT_SECONDS", "120") or "120"),
        upstream_api_key=upstream_api_key,
        upstream_api_key_header=env("UPSTREAM_API_KEY_HEADER", "Authorization") or "Authorization",
        upstream_api_key_prefix=env("UPSTREAM_API_KEY_PREFIX", "ApiKey") or "",
        upstream_extra_headers={str(k): str(v) for k, v in extra_headers_obj.items()},
        assistant_id=env("LANGGRAPH_SHIM_ASSISTANT_ID", "agent") or "agent",
        graph_id=env("LANGGRAPH_SHIM_GRAPH_ID", "agent") or "agent",
        assistant_name=env("LANGGRAPH_SHIM_NAME", "LangGraph Shim") or "LangGraph Shim",
        assistant_description=env(
            "LANGGRAPH_SHIM_DESCRIPTION",
            "Compatibility shim for non-LangGraph backends",
        )
        or "Compatibility shim for non-LangGraph backends",
    )


SETTINGS = load_settings()
STORE = LangGraphShimStore(SETTINGS.db_path)

app = FastAPI(title="LangGraph Shim", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class UpstreamError(RuntimeError):
    def __init__(self, message: str, status_code: int = 502, detail: Any | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


def sse_event(event: str, data: Any, event_id: int | None = None) -> str:
    lines: list[str] = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str)
    for line in payload.splitlines() or [""]:
        lines.append(f"data: {line}")
    return "\n".join(lines) + "\n\n"


def build_upstream_headers(settings: LangGraphShimSettings) -> dict[str, str]:
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/plain, */*",
    }
    if settings.upstream_extra_headers:
        headers.update(settings.upstream_extra_headers)
    if settings.upstream_api_key:
        key = settings.upstream_api_key
        header_name = settings.upstream_api_key_header
        if header_name.lower() == "authorization" and settings.upstream_api_key_prefix:
            headers[header_name] = f"{settings.upstream_api_key_prefix} {key}"
        elif settings.upstream_api_key_prefix:
            headers[header_name] = f"{settings.upstream_api_key_prefix} {key}"
        else:
            headers[header_name] = key
    return headers


def extract_prompt(input_values: Any, merged_values: dict[str, Any]) -> str:
    messages: list[Any] = []
    if isinstance(input_values, dict) and isinstance(input_values.get("messages"), list):
        messages = input_values["messages"]
    elif isinstance(merged_values.get("messages"), list):
        messages = merged_values["messages"]

    human = find_last_human_message(messages)
    if human is not None:
        prompt = content_to_text(human.get("content"))
        if prompt:
            return prompt.strip()

    if isinstance(input_values, dict):
        for key in ("message", "prompt", "text"):
            if isinstance(input_values.get(key), str) and input_values[key].strip():
                return input_values[key].strip()

    if isinstance(merged_values.get("messages"), list):
        text = extract_text(merged_values["messages"])
        if text:
            return text

    return "Please respond to the user's message."


async def call_upstream(
    settings: LangGraphShimSettings,
    payload: dict[str, Any],
) -> tuple[Any, str, int]:
    headers = build_upstream_headers(settings)
    async with httpx.AsyncClient(timeout=settings.upstream_timeout) as client:
        response = await client.request(
            settings.upstream_method,
            settings.upstream_url,
            json=payload,
            headers=headers,
        )

    body_text = response.text
    if response.status_code >= 400:
        raise UpstreamError(
            f"Upstream request failed with HTTP {response.status_code}",
            status_code=response.status_code,
            detail=body_text,
        )

    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type.lower():
        try:
            return response.json(), body_text, response.status_code
        except Exception:
            return body_text, body_text, response.status_code

    try:
        return response.json(), body_text, response.status_code
    except Exception:
        return body_text, body_text, response.status_code


def make_run_record(thread_id: str, assistant_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": uuid.uuid4().hex,
        "thread_id": thread_id,
        "assistant_id": assistant_id,
        "payload": payload,
    }


async def execute_run_core(
    *,
    store: LangGraphShimStore,
    settings: LangGraphShimSettings,
    thread_id: str,
    assistant_id: str,
    payload: dict[str, Any],
    run_id: str,
    record_event: Callable[[str, Any], Any],
) -> dict[str, Any]:
    thread = store.get_thread(thread_id)
    if thread is None:
        store.ensure_thread(
            thread_id,
            metadata=deepcopy(payload.get("metadata") or {}),
            assistant_id=assistant_id,
            graph_id=assistant_id,
            if_exists="ignore",
        )
    else:
        store.update_thread_metadata(
            thread_id,
            {"assistant_id": assistant_id, "graph_id": assistant_id},
        )

    store.set_thread_status(thread_id, "busy")

    checkpoint_filter = payload.get("checkpoint")
    base_state = store.get_state(thread_id, checkpoint_filter)
    base_values = deepcopy(base_state["values"])
    incoming_values = payload.get("input")
    merged_values = merge_values(base_values, incoming_values)
    prompt = extract_prompt(incoming_values, merged_values)

    upstream_payload: dict[str, Any] = {
        "message": prompt,
        "prompt": prompt,
        "messages": merged_values.get("messages", []),
        "input": incoming_values,
        "state": base_values,
        "thread_id": thread_id,
        "assistant_id": assistant_id,
        "checkpoint": checkpoint_filter,
        "config": payload.get("config"),
        "context": payload.get("context"),
        "metadata": payload.get("metadata"),
        "command": payload.get("command"),
        "stream_mode": payload.get("stream_mode"),
        "stream_subgraphs": payload.get("stream_subgraphs"),
        "stream_resumable": payload.get("stream_resumable"),
    }

    await record_event(
        "metadata",
        {"run_id": run_id, "thread_id": thread_id},
    )

    if store.is_run_cancel_requested(run_id):
        store.update_run(
            run_id,
            status="interrupted",
            error={"error": "CancelledError", "message": "Run was cancelled"},
        )
        store.set_thread_status(thread_id, "idle")
        return {
            "status": "interrupted",
            "run_id": run_id,
            "thread_id": thread_id,
            "assistant_id": assistant_id,
            "values": None,
            "checkpoint": None,
            "response": None,
            "error": {"error": "CancelledError", "message": "Run was cancelled"},
        }

    try:
        response_data, raw_text, _status_code = await call_upstream(settings, upstream_payload)
        if store.is_run_cancel_requested(run_id):
            store.update_run(
                run_id,
                status="interrupted",
                error={"error": "CancelledError", "message": "Run was cancelled"},
                response=response_data,
            )
            store.set_thread_status(thread_id, "idle")
            return {
                "status": "interrupted",
                "run_id": run_id,
                "thread_id": thread_id,
                "assistant_id": assistant_id,
                "values": None,
                "checkpoint": None,
                "response": response_data,
                "error": {"error": "CancelledError", "message": "Run was cancelled"},
            }

        assistant_message = build_assistant_message(response_data)
        if not isinstance(assistant_message.get("content"), (str, list)) or not assistant_message.get("content"):
            assistant_message["content"] = raw_text or "I couldn't generate a response."

        final_values = deepcopy(merged_values)
        final_messages = list(final_values.get("messages") or [])
        final_messages.append(assistant_message)
        final_values["messages"] = final_messages

        parent_checkpoint_id = base_state["checkpoint"]["checkpoint_id"]
        step = store.next_step(thread_id)
        checkpoint_metadata = {
            "source": "update",
            "step": step,
            "writes": {"messages": [assistant_message]},
        }

        checkpoint_state = store.append_checkpoint(
            thread_id,
            values=final_values,
            parent_checkpoint_id=parent_checkpoint_id,
            metadata=checkpoint_metadata,
            run_id=run_id,
            assistant_id=assistant_id,
            graph_id=assistant_id,
        )

        await record_event("messages", [assistant_message, {}])
        await record_event("values", final_values)
        await record_event(
            "checkpoints",
            {
                "values": final_values,
                "next": [],
                "config": {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": checkpoint_state["checkpoint"]["checkpoint_id"],
                    }
                },
                "metadata": checkpoint_metadata,
                "tasks": [],
            },
        )

        store.update_run(
            run_id,
            status="success",
            response=response_data,
            checkpoint_id=checkpoint_state["checkpoint"]["checkpoint_id"],
        )
        return {
            "status": "success",
            "run_id": run_id,
            "thread_id": thread_id,
            "assistant_id": assistant_id,
            "values": final_values,
            "checkpoint": checkpoint_state["checkpoint"],
            "response": response_data,
            "assistant_message": assistant_message,
            "error": None,
        }
    except UpstreamError as exc:
        error_payload = {
            "error": exc.__class__.__name__,
            "message": str(exc.detail or exc),
            "status_code": exc.status_code,
        }
        await record_event("error", error_payload)
        store.update_run(
            run_id,
            status="error",
            error=error_payload,
        )
        store.set_thread_status(thread_id, "error", error=error_payload)
        return {
            "status": "error",
            "run_id": run_id,
            "thread_id": thread_id,
            "assistant_id": assistant_id,
            "values": None,
            "checkpoint": None,
            "response": None,
            "error": error_payload,
        }
    except Exception as exc:
        error_payload = {
            "error": exc.__class__.__name__,
            "message": str(exc),
        }
        await record_event("error", error_payload)
        store.update_run(
            run_id,
            status="error",
            error=error_payload,
        )
        store.set_thread_status(thread_id, "error", error=error_payload)
        return {
            "status": "error",
            "run_id": run_id,
            "thread_id": thread_id,
            "assistant_id": assistant_id,
            "values": None,
            "checkpoint": None,
            "response": None,
            "error": error_payload,
        }


def thread_response(thread: dict[str, Any]) -> dict[str, Any]:
    return thread


def assistant_payload(settings: LangGraphShimSettings, assistant_id: str | None = None) -> dict[str, Any]:
    resolved_id = assistant_id or settings.assistant_id
    return {
        "assistant_id": resolved_id,
        "graph_id": settings.graph_id if assistant_id is None else resolved_id,
        "config": {},
        "context": {},
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "metadata": {},
        "version": 1,
        "name": settings.assistant_name,
        "description": settings.assistant_description,
    }


def assistant_graph_payload() -> dict[str, Any]:
    return {
        "nodes": [
            {
                "id": "chat",
                "name": "Chat",
                "metadata": {"type": "compatibility-shim"},
            }
        ],
        "edges": [],
    }


def assistant_schema_payload(assistant_id: str) -> dict[str, Any]:
    message_schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "type": {"type": "string"},
            "content": {},
        },
        "required": ["type", "content"],
        "additionalProperties": True,
    }
    return {
        "graph_id": assistant_id,
        "input_schema": {
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "items": message_schema,
                },
                "context": {"type": "object"},
            },
            "additionalProperties": True,
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "items": message_schema,
                }
            },
            "additionalProperties": True,
        },
        "state_schema": {
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "items": message_schema,
                },
                "ui": {"type": "array"},
            },
            "additionalProperties": True,
        },
        "config_schema": {"type": "object"},
        "context_schema": {"type": "object"},
    }


def parse_last_event_id(request: Request) -> int:
    header = request.headers.get("Last-Event-ID") or request.headers.get("last-event-id")
    if not header:
        return 0
    try:
        return max(int(header), 0)
    except Exception:
        return 0


@app.get("/")
@app.get("/health")
@app.get("/healthz")
@app.get("/ready")
@app.get("/info")
async def info() -> dict[str, Any]:
    return {
        "name": "langgraph-shim",
        "status": "ok",
        "time": now_iso(),
        "upstream_url": SETTINGS.upstream_url,
    }


@app.get("/assistants/{assistant_id}")
async def get_assistant(assistant_id: str) -> dict[str, Any]:
    return assistant_payload(SETTINGS, assistant_id)


@app.post("/assistants/search")
async def search_assistants(payload: dict[str, Any] | None = Body(default=None)) -> list[dict[str, Any]]:
    payload = payload or {}
    graph_id = payload.get("graph_id")
    metadata = payload.get("metadata")
    assistant = assistant_payload(SETTINGS)
    if graph_id is not None and graph_id != assistant["graph_id"]:
        return []
    if metadata is not None and not isinstance(metadata, dict):
        return []
    if isinstance(metadata, dict) and metadata:
        if not all(assistant["metadata"].get(key) == value for key, value in metadata.items()):
            return []
    return [assistant]


@app.get("/assistants/{assistant_id}/graph")
async def get_assistant_graph(assistant_id: str) -> dict[str, Any]:
    return assistant_graph_payload()


@app.get("/assistants/{assistant_id}/schemas")
async def get_assistant_schemas(assistant_id: str) -> dict[str, Any]:
    return assistant_schema_payload(assistant_id)


@app.post("/threads")
async def create_thread(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    payload = payload or {}
    thread_id = payload.get("thread_id") or payload.get("threadId") or uuid.uuid4().hex
    metadata = deepcopy(payload.get("metadata") or {})
    graph_id = metadata.get("graph_id") or payload.get("graph_id") or payload.get("graphId")
    assistant_id = metadata.get("assistant_id") or payload.get("assistant_id")
    if_exists = payload.get("if_exists") or payload.get("ifExists") or "raise"
    try:
        thread = STORE.ensure_thread(
            thread_id,
            metadata=metadata,
            assistant_id=assistant_id,
            graph_id=graph_id,
            if_exists=if_exists,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return thread_response(thread)


@app.get("/threads/{thread_id}")
async def get_thread(thread_id: str) -> dict[str, Any]:
    thread = STORE.get_thread(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    return thread_response(thread)


@app.patch("/threads/{thread_id}")
async def patch_thread(thread_id: str, payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    payload = payload or {}
    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise HTTPException(status_code=400, detail="metadata must be an object")
    try:
        return STORE.update_thread_metadata(thread_id, metadata)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Thread not found") from exc


@app.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str) -> Response:
    if STORE.get_thread(thread_id) is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    STORE.delete_thread(thread_id)
    return Response(status_code=204)


@app.post("/threads/search")
async def search_threads(payload: dict[str, Any] | None = Body(default=None)) -> list[dict[str, Any]]:
    payload = payload or {}
    return STORE.search_threads(
        metadata=payload.get("metadata"),
        ids=payload.get("ids"),
        limit=int(payload.get("limit") or 10),
        offset=int(payload.get("offset") or 0),
        status=payload.get("status"),
        sort_by=payload.get("sort_by"),
        sort_order=payload.get("sort_order"),
        select=payload.get("select"),
        values=payload.get("values"),
    )


@app.post("/threads/count")
async def count_threads(payload: dict[str, Any] | None = Body(default=None)) -> int:
    payload = payload or {}
    return STORE.count_threads(
        metadata=payload.get("metadata"),
        values=payload.get("values"),
        status=payload.get("status"),
    )


@app.post("/threads/prune")
async def prune_threads(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    payload = payload or {}
    thread_ids = payload.get("thread_ids") or []
    strategy = payload.get("strategy") or "delete"
    pruned = 0
    if strategy == "delete":
        for thread_id in thread_ids:
            if STORE.get_thread(thread_id) is not None:
                STORE.delete_thread(thread_id)
                pruned += 1
    return {"pruned_count": pruned}


@app.post("/threads/{thread_id}/copy")
async def copy_thread(thread_id: str) -> dict[str, Any]:
    source = STORE.get_thread(thread_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    new_thread_id = uuid.uuid4().hex
    copied = STORE.ensure_thread(
        new_thread_id,
        metadata=deepcopy(source["metadata"]),
        assistant_id=source["metadata"].get("assistant_id"),
        graph_id=source["metadata"].get("graph_id"),
        if_exists="raise",
    )
    return copied


@app.post("/threads/{thread_id}/state")
async def update_thread_state(thread_id: str, payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    payload = payload or {}
    values = payload.get("values")
    checkpoint = payload.get("checkpoint")
    checkpoint_id = payload.get("checkpoint_id") or payload.get("checkpointId")
    as_node = payload.get("as_node") or payload.get("asNode")
    try:
        return STORE.update_state(
            thread_id,
            values=values,
            checkpoint=checkpoint,
            checkpoint_id=checkpoint_id,
            as_node=as_node,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Thread not found") from exc


@app.post("/threads/{thread_id}/state/checkpoint")
async def get_state_for_checkpoint(thread_id: str, payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    payload = payload or {}
    checkpoint = payload.get("checkpoint")
    try:
        return STORE.get_state(thread_id, checkpoint)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Thread not found") from exc


@app.get("/threads/{thread_id}/state")
async def get_state(thread_id: str) -> dict[str, Any]:
    try:
        return STORE.get_state(thread_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Thread not found") from exc


@app.get("/threads/{thread_id}/state/{checkpoint_id}")
async def get_state_by_checkpoint_id(thread_id: str, checkpoint_id: str) -> dict[str, Any]:
    try:
        return STORE.get_state(thread_id, checkpoint_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Thread or checkpoint not found") from exc


@app.post("/threads/{thread_id}/history")
async def get_history(thread_id: str, payload: dict[str, Any] | None = Body(default=None)) -> list[dict[str, Any]]:
    payload = payload or {}
    try:
        return STORE.list_history(
            thread_id,
            limit=int(payload.get("limit") or 10),
            before=payload.get("before"),
            checkpoint=payload.get("checkpoint"),
            metadata=payload.get("metadata"),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Thread not found") from exc


@app.get("/threads/{thread_id}/runs")
async def list_runs(
    thread_id: str,
    limit: int = 10,
    offset: int = 0,
) -> list[dict[str, Any]]:
    return STORE.list_runs(thread_id, limit=limit, offset=offset)


@app.get("/threads/{thread_id}/runs/{run_id}")
async def get_run(thread_id: str, run_id: str) -> dict[str, Any]:
    run = STORE.get_run(run_id)
    if run is None or run["thread_id"] != thread_id:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.post("/threads/{thread_id}/runs/{run_id}/cancel")
async def cancel_run(
    thread_id: str,
    run_id: str,
    wait: str = "0",
    action: str = "interrupt",
) -> dict[str, Any]:
    run = STORE.get_run(run_id)
    if run is None or run["thread_id"] != thread_id:
        raise HTTPException(status_code=404, detail="Run not found")
    STORE.set_run_cancel_requested(run_id, True)
    STORE.update_run(
        run_id,
        status="interrupted",
        error={"error": "CancelledError", "message": f"Run cancelled with action={action}"},
    )
    STORE.set_thread_status(thread_id, "idle")
    return STORE.get_run(run_id) or {"run_id": run_id, "thread_id": thread_id, "status": "interrupted"}


@app.post("/threads/{thread_id}/runs/{run_id}/join")
async def join_run(thread_id: str, run_id: str) -> dict[str, Any]:
    run = STORE.get_run(run_id)
    if run is None or run["thread_id"] != thread_id:
        raise HTTPException(status_code=404, detail="Run not found")
    if run["status"] == "error":
        return {"__error__": json_load_error(run.get("error"))}
    if run["status"] == "interrupted":
        return {"__error__": {"error": "CancelledError", "message": "Run interrupted"}}
    thread = STORE.get_thread(thread_id)
    return thread["values"] if thread is not None else {}


def json_load_error(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return {"error": "RunError", "message": value}
    return {"error": "RunError", "message": str(value)}


@app.get("/threads/{thread_id}/stream")
async def thread_stream(thread_id: str, request: Request) -> StreamingResponse:
    run = None
    runs = STORE.list_runs(thread_id, limit=1, offset=0)
    if runs:
        run = STORE.get_run(runs[0]["run_id"])
    if run is None:
        async def empty_stream():
            if False:
                yield ""  # pragma: no cover

        return StreamingResponse(empty_stream(), media_type="text/event-stream")
    return await run_stream_events(thread_id, run["run_id"], request, replay_only=True)


async def run_stream_events(
    thread_id: str,
    run_id: str,
    request: Request,
    *,
    replay_only: bool = False,
) -> StreamingResponse:
    last_event_id = parse_last_event_id(request)

    async def event_source():
        current_event_id = last_event_id
        if replay_only:
            while True:
                events = STORE.get_run_events_since(run_id, current_event_id)
                if events:
                    for event in events:
                        current_event_id = int(event["event_id"])
                        yield sse_event(event["event"], event["data"], current_event_id)
                    continue
                run = STORE.get_run(run_id)
                if run is None or run["status"] in {"success", "error", "interrupted"}:
                    break
                try:
                    disconnected = await asyncio.wait_for(request.is_disconnected(), timeout=0.5)
                    if disconnected:
                        break
                except asyncio.TimeoutError:
                    continue
            return

        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        async def record_event(event: str, data: Any) -> None:
            event_id = STORE.append_run_event(run_id, event, data)
            await queue.put({"kind": "event", "event_id": event_id, "event": event, "data": data})

        async def producer() -> None:
            result = await execute_run_core(
                store=STORE,
                settings=SETTINGS,
                thread_id=thread_id,
                assistant_id=STORE.get_run(run_id)["assistant_id"] if STORE.get_run(run_id) else SETTINGS.assistant_id,
                payload=json_load_input_from_run(run_id),
                run_id=run_id,
                record_event=record_event,
            )
            await queue.put({"kind": "done", "result": result})

        task = asyncio.create_task(producer())
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    if await request.is_disconnected():
                        task.cancel()
                        break
                    continue
                if item["kind"] == "event":
                    current_event_id = int(item["event_id"])
                    yield sse_event(item["event"], item["data"], current_event_id)
                elif item["kind"] == "done":
                    break
        finally:
            if not task.done():
                task.cancel()

    headers = {
        "Content-Location": f"/threads/{thread_id}/runs/{run_id}",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers=headers,
    )


def json_load_input_from_run(run_id: str) -> dict[str, Any]:
    row = STORE.get_run_row(run_id)
    if row is None:
        return {}
    payload = json_loads(row["input_json"], {})
    return payload if isinstance(payload, dict) else {}


@app.post("/threads/{thread_id}/runs/stream")
async def stream_run(thread_id: str, request: Request, payload: dict[str, Any] | None = Body(default=None)) -> StreamingResponse:
    payload = payload or {}
    assistant_id = payload.get("assistant_id") or SETTINGS.assistant_id
    run_id = uuid.uuid4().hex
    if STORE.get_thread(thread_id) is None:
        STORE.ensure_thread(
            thread_id,
            metadata=deepcopy(payload.get("metadata") or {}),
            assistant_id=assistant_id,
            graph_id=assistant_id,
            if_exists="ignore",
        )
    STORE.create_run(
        run_id=run_id,
        thread_id=thread_id,
        assistant_id=assistant_id,
        payload=payload,
    )
    STORE.update_thread_metadata(thread_id, {"assistant_id": assistant_id, "graph_id": assistant_id})

    async def event_source():
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        async def record_event(event: str, data: Any) -> None:
            event_id = STORE.append_run_event(run_id, event, data)
            await queue.put({"kind": "event", "event_id": event_id, "event": event, "data": data})

        async def producer() -> None:
            result = await execute_run_core(
                store=STORE,
                settings=SETTINGS,
                thread_id=thread_id,
                assistant_id=assistant_id,
                payload=payload,
                run_id=run_id,
                record_event=record_event,
            )
            await queue.put({"kind": "done", "result": result})

        task = asyncio.create_task(producer())
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    if await request.is_disconnected():
                        task.cancel()
                        break
                    continue
                if item["kind"] == "event":
                    yield sse_event(item["event"], item["data"], int(item["event_id"]))
                elif item["kind"] == "done":
                    break
        finally:
            if not task.done():
                task.cancel()

    headers = {
        "Content-Location": f"/threads/{thread_id}/runs/{run_id}",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers=headers,
    )


@app.post("/runs/stream")
async def stream_stateless_run(request: Request, payload: dict[str, Any] | None = Body(default=None)) -> StreamingResponse:
    payload = payload or {}
    thread_id = uuid.uuid4().hex
    assistant_id = payload.get("assistant_id") or SETTINGS.assistant_id
    STORE.ensure_thread(
        thread_id,
        metadata=deepcopy(payload.get("metadata") or {}),
        assistant_id=assistant_id,
        graph_id=assistant_id,
        if_exists="ignore",
    )
    return await stream_run(thread_id, request, payload)


@app.post("/threads/{thread_id}/runs")
async def create_run(thread_id: str, payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    payload = payload or {}
    assistant_id = payload.get("assistant_id") or SETTINGS.assistant_id
    run_id = uuid.uuid4().hex
    if STORE.get_thread(thread_id) is None:
        STORE.ensure_thread(
            thread_id,
            metadata=deepcopy(payload.get("metadata") or {}),
            assistant_id=assistant_id,
            graph_id=assistant_id,
            if_exists="ignore",
        )
    STORE.create_run(run_id=run_id, thread_id=thread_id, assistant_id=assistant_id, payload=payload)
    STORE.update_thread_metadata(thread_id, {"assistant_id": assistant_id, "graph_id": assistant_id})

    async def record_event(event: str, data: Any) -> None:
        STORE.append_run_event(run_id, event, data)

    result = await execute_run_core(
        store=STORE,
        settings=SETTINGS,
        thread_id=thread_id,
        assistant_id=assistant_id,
        payload=payload,
        run_id=run_id,
        record_event=record_event,
    )
    return STORE.get_run(run_id) or {
        "run_id": run_id,
        "thread_id": thread_id,
        "assistant_id": assistant_id,
        "status": result["status"],
    }


@app.post("/runs")
async def create_stateless_run(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    payload = payload or {}
    thread_id = uuid.uuid4().hex
    return await create_run(thread_id, payload)


@app.post("/threads/{thread_id}/runs/wait")
async def wait_run(thread_id: str, payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    payload = payload or {}
    assistant_id = payload.get("assistant_id") or SETTINGS.assistant_id
    run_id = uuid.uuid4().hex
    if STORE.get_thread(thread_id) is None:
        STORE.ensure_thread(
            thread_id,
            metadata=deepcopy(payload.get("metadata") or {}),
            assistant_id=assistant_id,
            graph_id=assistant_id,
            if_exists="ignore",
        )
    STORE.create_run(run_id=run_id, thread_id=thread_id, assistant_id=assistant_id, payload=payload)
    STORE.update_thread_metadata(thread_id, {"assistant_id": assistant_id, "graph_id": assistant_id})

    async def record_event(event: str, data: Any) -> None:
        STORE.append_run_event(run_id, event, data)

    result = await execute_run_core(
        store=STORE,
        settings=SETTINGS,
        thread_id=thread_id,
        assistant_id=assistant_id,
        payload=payload,
        run_id=run_id,
        record_event=record_event,
    )
    if result["status"] == "error":
        return {"__error__": result["error"]}
    if result["status"] == "interrupted":
        return {"__error__": result["error"]}
    return result["values"] or {}


@app.post("/runs/wait")
async def wait_stateless_run(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
    payload = payload or {}
    thread_id = uuid.uuid4().hex
    STORE.ensure_thread(
        thread_id,
        metadata=deepcopy(payload.get("metadata") or {}),
        assistant_id=payload.get("assistant_id") or SETTINGS.assistant_id,
        graph_id=payload.get("assistant_id") or SETTINGS.assistant_id,
        if_exists="ignore",
    )
    return await wait_run(thread_id, payload)


@app.get("/runs/{run_id}/stream")
async def join_stateless_run_stream(run_id: str, request: Request) -> StreamingResponse:
    run = STORE.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return await run_stream_events(run["thread_id"], run_id, request, replay_only=True)


@app.get("/threads/{thread_id}/runs/{run_id}/stream")
async def join_thread_run_stream(thread_id: str, run_id: str, request: Request) -> StreamingResponse:
    run = STORE.get_run(run_id)
    if run is None or run["thread_id"] != thread_id:
        raise HTTPException(status_code=404, detail="Run not found")
    return await run_stream_events(thread_id, run_id, request, replay_only=True)


@app.get("/threads/{thread_id}/runs/{run_id}/join")
async def join_thread_run(thread_id: str, run_id: str) -> dict[str, Any]:
    run = STORE.get_run(run_id)
    if run is None or run["thread_id"] != thread_id:
        raise HTTPException(status_code=404, detail="Run not found")
    if run["status"] == "error":
        return {"__error__": json_load_error(run.get("error"))}
    if run["status"] == "interrupted":
        return {"__error__": {"error": "CancelledError", "message": "Run interrupted"}}
    thread = STORE.get_thread(thread_id)
    return thread["values"] if thread is not None else {}


@app.get("/runs/{run_id}")
async def get_stateless_run(run_id: str) -> dict[str, Any]:
    run = STORE.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.delete("/threads/{thread_id}/runs/{run_id}")
async def delete_run(thread_id: str, run_id: str) -> Response:
    run = STORE.get_run(run_id)
    if run is None or run["thread_id"] != thread_id:
        raise HTTPException(status_code=404, detail="Run not found")
    # SQLite cascade handles events if the run row is removed.
    with STORE._lock:  # noqa: SLF001 - local persistence helper
        with STORE._connect() as conn:  # noqa: SLF001 - local persistence helper
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
    return Response(status_code=204)


def json_load_error(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return {"error": "RunError", "message": value}
    return {"error": "RunError", "message": str(value)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(env("PORT", "8080") or "8080"),
        reload=False,
    )
