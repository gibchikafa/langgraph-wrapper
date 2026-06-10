# langgraph-wrapper

# LangGraph Shim

This directory contains a small LangGraph-compatible HTTP service that wraps a
plain upstream chat endpoint.

The goal is to keep `agent-chat-ui` unchanged. The UI still talks to a
LangGraph-shaped API, while this shim translates those requests into a simple
HTTP call to your backend.

## What It Implements

- `GET /info`
- `GET /health`
- `POST /threads`
- `POST /threads/search`
- `GET /threads/{thread_id}`
- `GET /threads/{thread_id}/state`
- `POST /threads/{thread_id}/history`
- `POST /threads/{thread_id}/runs/stream`
- `POST /threads/{thread_id}/runs`
- `POST /threads/{thread_id}/runs/wait`
- `GET /threads/{thread_id}/runs/{run_id}/stream`
- `GET /threads/{thread_id}/runs/{run_id}/join`
- `POST /threads/{thread_id}/runs/{run_id}/cancel`

It also stores thread history, runs, and checkpoints in SQLite so branch history
still works after a reload if the SQLite file is persisted.

## Upstream Contract

The shim sends a JSON envelope like this to `UPSTREAM_CHAT_URL`:

```json
{
  "message": "latest human prompt",
  "prompt": "latest human prompt",
  "messages": [],
  "input": {},
  "state": {},
  "thread_id": "thread-id",
  "assistant_id": "assistant-id",
  "checkpoint": null,
  "config": {},
  "context": {},
  "metadata": {},
  "command": null,
  "stream_mode": ["values"],
  "stream_subgraphs": true,
  "stream_resumable": true
}
```

Your upstream service can ignore fields it does not need. The shim looks for an
assistant reply in common response fields such as:

- `response`
- `answer`
- `message`
- `output`
- `text`
- `content`
- `reply`
- `result`

If the response is JSON with a `messages` list, it will use the last assistant
message in that list.

## Environment Variables

Copy `.env.example` and adjust it for your backend.

Important variables:

- `UPSTREAM_CHAT_URL`
- `UPSTREAM_CHAT_METHOD`
- `UPSTREAM_API_KEY`
- `UPSTREAM_API_KEY_HEADER`
- `UPSTREAM_API_KEY_PREFIX`
- `UPSTREAM_EXTRA_HEADERS_JSON`
- `LANGGRAPH_SHIM_DB_PATH`

The shim also falls back to `LANGSMITH_API_KEY`, `LANGGRAPH_API_KEY`, and
`LANGCHAIN_API_KEY` if `UPSTREAM_API_KEY` is not set.

## Hopsworks Usage

1. Deploy `app.py` as the Python app entrypoint.
2. Set the env vars from `.env.example`.
3. Point `agent-chat-ui` at the Hopsworks app URL for `NEXT_PUBLIC_API_URL`.
4. Keep `NEXT_PUBLIC_ASSISTANT_ID` aligned with `LANGGRAPH_SHIM_ASSISTANT_ID`.

## In-Process Agent Mode

If you want to run the shim and the agent logic in the same pod, install the
package into the `python-agent-pipeline` image and use the shim as a library:

1. Add `langgraph-shim` to `docker-images/base-image/python-agent-pipeline/requirements.txt`.
2. Import `app` and `set_upstream_handler` from `langgraph_shim.app`.
3. Set the handler to your local `predict()` coroutine before starting Uvicorn.

See [`examples/agent_with_shim.py`](/Users/gibson/Work/langgraph-shim/examples/agent_with_shim.py)
for a complete template.

If the upstream backend is behind Istio and requires an API key, use:

```bash
UPSTREAM_API_KEY=<key>
UPSTREAM_API_KEY_HEADER=Authorization
UPSTREAM_API_KEY_PREFIX=ApiKey
```

If the backend expects `X-Api-Key`, set:

```bash
UPSTREAM_API_KEY_HEADER=X-Api-Key
UPSTREAM_API_KEY_PREFIX=
```

## Local Run

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8080
```

## Upstream Agent Example

If you want a ready-made upstream service to place behind the shim, use:

- [`examples/upstream_agent.py`](/Users/gibson/Work/langgraph-shim/examples/upstream_agent.py)
- [`examples/requirements.txt`](/Users/gibson/Work/langgraph-shim/examples/requirements.txt)
- [`examples/.env.example`](/Users/gibson/Work/langgraph-shim/examples/.env.example)

Deploy that app in Hopsworks, then point `UPSTREAM_CHAT_URL` on the shim to the
agent's `/chat` endpoint. Keep `NEXT_PUBLIC_API_URL` in `agent-chat-ui`
pointing at the shim, not the agent.
