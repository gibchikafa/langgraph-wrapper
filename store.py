from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def json_loads(text: str | None, default: Any = None) -> Any:
    if not text:
        return deepcopy(default)
    try:
        return json.loads(text)
    except Exception:
        return deepcopy(default)


def is_subset(query: Any, value: Any) -> bool:
    if query is None:
        return True
    if isinstance(query, dict):
        if not isinstance(value, dict):
            return False
        for key, qv in query.items():
            if key not in value or not is_subset(qv, value[key]):
                return False
        return True
    if isinstance(query, list):
        return isinstance(value, list) and value == query
    return value == query


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif isinstance(block.get("text"), str):
                    parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(part for part in parts if part).strip()
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        if isinstance(content.get("content"), str):
            return content["content"]
    return ""


def find_last_human_message(messages: list[Any]) -> dict[str, Any] | None:
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("type") in {"human", "user"}:
            return message
    return None


def extract_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload.strip()

    if isinstance(payload, list):
        for item in reversed(payload):
            text = extract_text(item)
            if text:
                return text
        return ""

    if not isinstance(payload, dict):
        return ""

    for key in ("response", "answer", "message", "output", "text", "content", "reply", "result"):
        value = payload.get(key)
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            text = extract_text(value)
            if text:
                return text
        if isinstance(value, list):
            text = extract_text(value)
            if text:
                return text

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message")
            if isinstance(message, dict):
                text = extract_text(message)
                if text:
                    return text

    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("type") in {"ai", "assistant"}:
                text = extract_text(message)
                if text:
                    return text
    return ""


def build_assistant_message(payload: Any) -> dict[str, Any]:
    message: dict[str, Any] = {}

    if isinstance(payload, dict):
        candidate = payload.get("message")
        if isinstance(candidate, dict):
            message = deepcopy(candidate)
        else:
            messages = payload.get("messages")
            if isinstance(messages, list):
                for item in reversed(messages):
                    if isinstance(item, dict) and item.get("type") in {"ai", "assistant"}:
                        message = deepcopy(item)
                        break
        if not message:
            text = extract_text(payload)
            if text:
                message = {"type": "ai", "content": text}
    elif isinstance(payload, str):
        message = {"type": "ai", "content": payload}

    if not message:
        message = {"type": "ai", "content": "I couldn't generate a response."}

    message.setdefault("type", "ai")
    if message["type"] == "assistant":
        message["type"] = "ai"
    if not isinstance(message.get("content"), (str, list)):
        text = extract_text(payload)
        if text:
            message["content"] = text
        else:
            message["content"] = "I couldn't generate a response."
    if not message.get("id"):
        message["id"] = uuid.uuid4().hex
    return message


def build_checkpoint_object(
    thread_id: str,
    checkpoint_id: str,
    checkpoint_ns: str = "",
    checkpoint_map: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "thread_id": thread_id,
        "checkpoint_ns": checkpoint_ns,
        "checkpoint_id": checkpoint_id,
        "checkpoint_map": checkpoint_map or {},
    }


def build_state_payload(
    *,
    values: Any,
    checkpoint: dict[str, Any],
    parent_checkpoint: dict[str, Any] | None,
    metadata: Any,
    created_at: str,
    tasks: list[Any] | None = None,
    next_nodes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "values": values,
        "next": next_nodes or [],
        "checkpoint": checkpoint,
        "metadata": metadata,
        "created_at": created_at,
        "parent_checkpoint": parent_checkpoint,
        "tasks": tasks or [],
    }


def normalize_message(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        result = deepcopy(message)
        if result.get("type") == "assistant":
            result["type"] = "ai"
        if not result.get("id"):
            result["id"] = uuid.uuid4().hex
        return result
    if isinstance(message, str):
        return {"id": uuid.uuid4().hex, "type": "human", "content": message}
    return {"id": uuid.uuid4().hex, "type": "human", "content": str(message)}


def merge_values(base_values: Any, input_values: Any) -> dict[str, Any]:
    merged: dict[str, Any] = deepcopy(base_values or {})
    if not isinstance(merged, dict):
        merged = {}

    if input_values is None:
        merged.setdefault("messages", [])
        return merged

    if isinstance(input_values, str):
        input_values = {"messages": [{"type": "human", "content": input_values}]}
    elif not isinstance(input_values, dict):
        input_values = {"messages": [input_values]}

    for key, value in input_values.items():
        if key == "messages":
            current_messages = list(merged.get("messages") or [])
            if isinstance(value, list):
                current_messages.extend(
                    normalize_message(message) for message in value if message is not None
                )
            else:
                if value is not None:
                    current_messages.append(normalize_message(value))
            merged["messages"] = current_messages
        else:
            merged[key] = deepcopy(value)

    merged.setdefault("messages", [])
    return merged


@dataclass
class LangGraphShimSettings:
    db_path: str
    upstream_url: str
    upstream_method: str = "POST"
    upstream_timeout: float = 120.0
    upstream_api_key: str | None = None
    upstream_api_key_header: str = "Authorization"
    upstream_api_key_prefix: str = "ApiKey"
    upstream_extra_headers: dict[str, str] | None = None
    assistant_id: str = "agent"
    graph_id: str = "agent"
    assistant_name: str = "LangGraph Shim"
    assistant_description: str = "Compatibility shim for non-LangGraph backends"
    response_keys: tuple[str, ...] = (
        "response",
        "answer",
        "message",
        "output",
        "text",
        "content",
        "reply",
        "result",
    )


class LangGraphShimStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS threads (
                        thread_id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        state_updated_at TEXT NOT NULL,
                        status TEXT NOT NULL,
                        metadata_json TEXT NOT NULL,
                        values_json TEXT NOT NULL,
                        config_json TEXT,
                        error_json TEXT,
                        head_checkpoint_id TEXT,
                        assistant_id TEXT,
                        graph_id TEXT
                    );

                    CREATE TABLE IF NOT EXISTS checkpoints (
                        checkpoint_id TEXT PRIMARY KEY,
                        thread_id TEXT NOT NULL,
                        parent_checkpoint_id TEXT,
                        checkpoint_ns TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        step INTEGER NOT NULL,
                        values_json TEXT NOT NULL,
                        metadata_json TEXT NOT NULL,
                        next_json TEXT NOT NULL,
                        tasks_json TEXT NOT NULL,
                        run_id TEXT,
                        FOREIGN KEY(thread_id) REFERENCES threads(thread_id) ON DELETE CASCADE
                    );

                    CREATE TABLE IF NOT EXISTS runs (
                        run_id TEXT PRIMARY KEY,
                        thread_id TEXT NOT NULL,
                        assistant_id TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        status TEXT NOT NULL,
                        metadata_json TEXT NOT NULL,
                        input_json TEXT NOT NULL,
                        response_json TEXT,
                        error_json TEXT,
                        checkpoint_id TEXT,
                        cancel_requested INTEGER NOT NULL DEFAULT 0,
                        FOREIGN KEY(thread_id) REFERENCES threads(thread_id) ON DELETE CASCADE
                    );

                    CREATE TABLE IF NOT EXISTS run_events (
                        run_id TEXT NOT NULL,
                        event_id INTEGER NOT NULL,
                        event TEXT NOT NULL,
                        data_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY (run_id, event_id),
                        FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                    );
                    """
                )

    def _row_to_thread(self, row: sqlite3.Row) -> dict[str, Any]:
        metadata = json_loads(row["metadata_json"], {})
        assistant_id = row["assistant_id"] or metadata.get("assistant_id")
        graph_id = row["graph_id"] or metadata.get("graph_id")
        if assistant_id is not None:
            metadata.setdefault("assistant_id", assistant_id)
        if graph_id is not None:
            metadata.setdefault("graph_id", graph_id)
        return {
            "thread_id": row["thread_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "state_updated_at": row["state_updated_at"],
            "metadata": metadata,
            "status": row["status"],
            "values": json_loads(row["values_json"], {"messages": []}),
            "interrupts": {},
            "config": json_loads(row["config_json"], None),
            "error": json_loads(row["error_json"], None),
        }

    def _row_to_checkpoint(self, row: sqlite3.Row) -> dict[str, Any]:
        return build_checkpoint_object(
            row["thread_id"],
            row["checkpoint_id"],
            row["checkpoint_ns"],
            {},
        )

    def _row_to_state(self, row: sqlite3.Row) -> dict[str, Any]:
        parent = self.get_checkpoint_row(row["thread_id"], row["parent_checkpoint_id"])
        return build_state_payload(
            values=json_loads(row["values_json"], {"messages": []}),
            checkpoint=self._row_to_checkpoint(row),
            parent_checkpoint=self._row_to_checkpoint(parent) if parent else None,
            metadata=json_loads(row["metadata_json"], {}),
            created_at=row["created_at"],
            tasks=json_loads(row["tasks_json"], []),
            next_nodes=json_loads(row["next_json"], []),
        )

    def get_thread_row(self, thread_id: str) -> sqlite3.Row | None:
        with self._lock:
            with self._connect() as conn:
                return conn.execute(
                    "SELECT * FROM threads WHERE thread_id = ?",
                    (thread_id,),
                ).fetchone()

    def get_thread(self, thread_id: str) -> dict[str, Any] | None:
        row = self.get_thread_row(thread_id)
        if row is None:
            return None
        return self._row_to_thread(row)

    def ensure_thread(
        self,
        thread_id: str,
        *,
        metadata: dict[str, Any] | None = None,
        assistant_id: str | None = None,
        graph_id: str | None = None,
        if_exists: str = "raise",
    ) -> dict[str, Any]:
        existing = self.get_thread(thread_id)
        if existing is not None:
            if if_exists in {"ignore", "do_nothing", "keep"}:
                return existing
            if if_exists == "replace":
                self.delete_thread(thread_id)
            else:
                raise ValueError(f"Thread {thread_id} already exists")

        created_at = now_iso()
        metadata = deepcopy(metadata or {})
        if assistant_id is not None:
            metadata.setdefault("assistant_id", assistant_id)
        if graph_id is not None:
            metadata.setdefault("graph_id", graph_id)
        values = {"messages": []}

        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO threads (
                        thread_id, created_at, updated_at, state_updated_at,
                        status, metadata_json, values_json, config_json,
                        error_json, head_checkpoint_id, assistant_id, graph_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        thread_id,
                        created_at,
                        created_at,
                        created_at,
                        "idle",
                        json_dumps(metadata),
                        json_dumps(values),
                        None,
                        None,
                        None,
                        metadata.get("assistant_id"),
                        metadata.get("graph_id"),
                    ),
                )
                root_checkpoint_id = f"{thread_id}:root"
                conn.execute(
                    """
                    INSERT INTO checkpoints (
                        checkpoint_id, thread_id, parent_checkpoint_id,
                        checkpoint_ns, created_at, step, values_json,
                        metadata_json, next_json, tasks_json, run_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        root_checkpoint_id,
                        thread_id,
                        None,
                        "",
                        created_at,
                        0,
                        json_dumps(values),
                        json_dumps({"source": "input", "step": 0, "writes": {}}),
                        json_dumps([]),
                        json_dumps([]),
                        None,
                    ),
                )
                conn.execute(
                    """
                    UPDATE threads
                    SET head_checkpoint_id = ?, values_json = ?, state_updated_at = ?, updated_at = ?
                    WHERE thread_id = ?
                    """,
                    (
                        root_checkpoint_id,
                        json_dumps(values),
                        created_at,
                        created_at,
                        thread_id,
                    ),
                )
        return self.get_thread(thread_id) or {}

    def delete_thread(self, thread_id: str) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))

    def update_thread_metadata(self, thread_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        row = self.get_thread_row(thread_id)
        if row is None:
            raise KeyError(thread_id)
        metadata = json_loads(row["metadata_json"], {})
        metadata.update(deepcopy(patch))
        assistant_id = metadata.get("assistant_id")
        graph_id = metadata.get("graph_id")
        updated_at = now_iso()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE threads
                    SET metadata_json = ?, assistant_id = ?, graph_id = ?, updated_at = ?
                    WHERE thread_id = ?
                    """,
                    (json_dumps(metadata), assistant_id, graph_id, updated_at, thread_id),
                )
        return self.get_thread(thread_id) or {}

    def set_thread_status(
        self,
        thread_id: str,
        status: str,
        *,
        error: Any | None = None,
        values: Any | None = None,
        head_checkpoint_id: str | None = None,
    ) -> None:
        row = self.get_thread_row(thread_id)
        if row is None:
            raise KeyError(thread_id)
        updated_at = now_iso()
        state_updated_at = updated_at if values is not None else row["state_updated_at"]
        values_json = json_dumps(values) if values is not None else row["values_json"]
        if error is not None:
            error_json = json_dumps(error)
        elif status == "error":
            error_json = row["error_json"]
        else:
            error_json = None
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE threads
                    SET status = ?, updated_at = ?, state_updated_at = ?, values_json = ?,
                        error_json = ?, head_checkpoint_id = COALESCE(?, head_checkpoint_id)
                    WHERE thread_id = ?
                    """,
                    (
                        status,
                        updated_at,
                        state_updated_at,
                        values_json,
                        error_json,
                        head_checkpoint_id,
                        thread_id,
                    ),
                )

    def get_checkpoint_row(self, thread_id: str, checkpoint_id: str | None) -> sqlite3.Row | None:
        if checkpoint_id is None:
            return None
        with self._lock:
            with self._connect() as conn:
                return conn.execute(
                    """
                    SELECT * FROM checkpoints
                    WHERE thread_id = ? AND checkpoint_id = ?
                    """,
                    (thread_id, checkpoint_id),
                ).fetchone()

    def get_latest_checkpoint_row_for_namespace(
        self, thread_id: str, checkpoint_ns: str
    ) -> sqlite3.Row | None:
        with self._lock:
            with self._connect() as conn:
                return conn.execute(
                    """
                    SELECT * FROM checkpoints
                    WHERE thread_id = ? AND checkpoint_ns = ?
                    ORDER BY step DESC
                    LIMIT 1
                    """,
                    (thread_id, checkpoint_ns),
                ).fetchone()

    def get_head_checkpoint_row(self, thread_id: str) -> sqlite3.Row | None:
        row = self.get_thread_row(thread_id)
        if row is None:
            return None
        head_checkpoint_id = row["head_checkpoint_id"]
        if head_checkpoint_id:
            return self.get_checkpoint_row(thread_id, head_checkpoint_id)
        return None

    def get_state(self, thread_id: str, checkpoint: Any | None = None) -> dict[str, Any]:
        row = self.get_thread_row(thread_id)
        if row is None:
            raise KeyError(thread_id)

        checkpoint_row = None
        if isinstance(checkpoint, dict):
            checkpoint_id = checkpoint.get("checkpoint_id")
            checkpoint_ns = checkpoint.get("checkpoint_ns")
            if isinstance(checkpoint_id, str) and checkpoint_id:
                checkpoint_row = self.get_checkpoint_row(thread_id, checkpoint_id)
            elif isinstance(checkpoint_ns, str) and checkpoint_ns:
                checkpoint_row = self.get_latest_checkpoint_row_for_namespace(thread_id, checkpoint_ns)
        elif isinstance(checkpoint, str) and checkpoint:
            checkpoint_row = self.get_checkpoint_row(thread_id, checkpoint)

        if checkpoint_row is None:
            checkpoint_row = self.get_head_checkpoint_row(thread_id)
        if checkpoint_row is None:
            raise KeyError(thread_id)
        return self._row_to_state(checkpoint_row)

    def list_history(
        self,
        thread_id: str,
        *,
        limit: int = 10,
        before: Any | None = None,
        checkpoint: Any | None = None,
        metadata: Any | None = None,
    ) -> list[dict[str, Any]]:
        row = self.get_thread_row(thread_id)
        if row is None:
            raise KeyError(thread_id)

        checkpoint_ns = None
        checkpoint_id = None
        checkpoint_step = None

        if isinstance(checkpoint, dict):
            checkpoint_ns = (
                checkpoint.get("checkpoint_ns")
                if isinstance(checkpoint.get("checkpoint_ns"), str)
                else None
            )
            checkpoint_id = (
                checkpoint.get("checkpoint_id")
                if isinstance(checkpoint.get("checkpoint_id"), str)
                else None
            )
        elif isinstance(checkpoint, str):
            checkpoint_id = checkpoint

        before_id = None
        if isinstance(before, dict):
            before_id = before.get("checkpoint_id") if isinstance(before.get("checkpoint_id"), str) else None
            if before_id is None and isinstance(before.get("checkpoint_ns"), str):
                before_ns_row = self.get_latest_checkpoint_row_for_namespace(thread_id, before["checkpoint_ns"])
                if before_ns_row is not None:
                    before_id = before_ns_row["parent_checkpoint_id"]
        elif isinstance(before, str):
            before_id = before

        if before_id is not None:
            before_row = self.get_checkpoint_row(thread_id, before_id)
            if before_row is not None:
                checkpoint_step = int(before_row["step"])

        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT * FROM checkpoints
                    WHERE thread_id = ?
                    ORDER BY step DESC
                    """,
                    (thread_id,),
                ).fetchall()

        results: list[dict[str, Any]] = []
        for checkpoint_row in rows:
            if checkpoint_ns is not None and checkpoint_row["checkpoint_ns"] != checkpoint_ns:
                continue
            if checkpoint_id is not None and checkpoint_row["checkpoint_id"] != checkpoint_id:
                continue
            if checkpoint_step is not None and int(checkpoint_row["step"]) >= checkpoint_step:
                continue
            state = self._row_to_state(checkpoint_row)
            if metadata is not None and not is_subset(metadata, state.get("metadata")):
                continue
            results.append(state)
            if len(results) >= max(limit, 0):
                break
        return results

    def next_step(self, thread_id: str) -> int:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT COALESCE(MAX(step), 0) + 1 AS next_step FROM checkpoints WHERE thread_id = ?",
                    (thread_id,),
                ).fetchone()
                return int(row["next_step"])

    def append_checkpoint(
        self,
        thread_id: str,
        *,
        values: Any,
        parent_checkpoint_id: str | None,
        metadata: dict[str, Any] | None = None,
        run_id: str | None = None,
        checkpoint_ns: str = "",
        next_nodes: list[str] | None = None,
        tasks: list[Any] | None = None,
        assistant_id: str | None = None,
        graph_id: str | None = None,
    ) -> dict[str, Any]:
        row = self.get_thread_row(thread_id)
        if row is None:
            raise KeyError(thread_id)

        created_at = now_iso()
        checkpoint_id = uuid.uuid4().hex
        step = self.next_step(thread_id)
        metadata = deepcopy(metadata or {})
        checkpoint_obj = build_checkpoint_object(thread_id, checkpoint_id, checkpoint_ns, {})
        parent_row = self.get_checkpoint_row(thread_id, parent_checkpoint_id)
        parent_obj = self._row_to_checkpoint(parent_row) if parent_row else None

        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO checkpoints (
                        checkpoint_id, thread_id, parent_checkpoint_id,
                        checkpoint_ns, created_at, step, values_json,
                        metadata_json, next_json, tasks_json, run_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        checkpoint_id,
                        thread_id,
                        parent_checkpoint_id,
                        checkpoint_ns,
                        created_at,
                        step,
                        json_dumps(values),
                        json_dumps(metadata),
                        json_dumps(next_nodes or []),
                        json_dumps(tasks or []),
                        run_id,
                    ),
                )
                merged_metadata = json_loads(row["metadata_json"], {})
                if assistant_id is not None:
                    merged_metadata["assistant_id"] = assistant_id
                if graph_id is not None:
                    merged_metadata["graph_id"] = graph_id
                conn.execute(
                    """
                    UPDATE threads
                    SET updated_at = ?, state_updated_at = ?, status = ?, values_json = ?,
                        error_json = NULL, head_checkpoint_id = ?, metadata_json = ?, assistant_id = ?, graph_id = ?
                    WHERE thread_id = ?
                    """,
                    (
                        created_at,
                        created_at,
                        "idle",
                        json_dumps(values),
                        checkpoint_id,
                        json_dumps(merged_metadata),
                        merged_metadata.get("assistant_id"),
                        merged_metadata.get("graph_id"),
                        thread_id,
                    ),
                )
        return build_state_payload(
            values=values,
            checkpoint=checkpoint_obj,
            parent_checkpoint=parent_obj,
            metadata=metadata,
            created_at=created_at,
            tasks=tasks or [],
            next_nodes=next_nodes or [],
        )

    def update_state(
        self,
        thread_id: str,
        *,
        values: Any,
        checkpoint: Any | None = None,
        checkpoint_id: str | None = None,
        as_node: str | None = None,
        assistant_id: str | None = None,
        graph_id: str | None = None,
    ) -> dict[str, Any]:
        base_state = self.get_state(thread_id, checkpoint if checkpoint is not None else None)
        base_values = base_state["values"]
        merged_values = merge_values(base_values, values)
        parent_checkpoint_id = checkpoint_id
        if parent_checkpoint_id is None and isinstance(checkpoint, dict):
            parent_checkpoint_id = checkpoint.get("checkpoint_id")
        if parent_checkpoint_id is None:
            parent_checkpoint_id = base_state["checkpoint"]["checkpoint_id"]
        metadata = {"source": "update", "step": self.next_step(thread_id), "writes": {}}
        if as_node:
            metadata["as_node"] = as_node
        return self.append_checkpoint(
            thread_id,
            values=merged_values,
            parent_checkpoint_id=parent_checkpoint_id,
            metadata=metadata,
            assistant_id=assistant_id,
            graph_id=graph_id,
        )

    def patch_state_metadata(self, thread_id: str, metadata: dict[str, Any]) -> None:
        self.update_thread_metadata(thread_id, metadata)

    def create_run(
        self,
        *,
        run_id: str,
        thread_id: str,
        assistant_id: str,
        payload: dict[str, Any],
    ) -> None:
        created_at = now_iso()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO runs (
                        run_id, thread_id, assistant_id, created_at, updated_at,
                        status, metadata_json, input_json, response_json,
                        error_json, checkpoint_id, cancel_requested
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                    """,
                    (
                        run_id,
                        thread_id,
                        assistant_id,
                        created_at,
                        created_at,
                        "running",
                        json_dumps(payload.get("metadata") or {}),
                        json_dumps(payload.get("input")),
                        None,
                        None,
                        None,
                    ),
                )

    def get_run_row(self, run_id: str) -> sqlite3.Row | None:
        with self._lock:
            with self._connect() as conn:
                return conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self.get_run_row(run_id)
        if row is None:
            return None
        return {
            "run_id": row["run_id"],
            "thread_id": row["thread_id"],
            "assistant_id": row["assistant_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "status": row["status"],
            "metadata": json_loads(row["metadata_json"], {}),
        }

    def list_runs(self, thread_id: str, *, limit: int = 10, offset: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT * FROM runs
                    WHERE thread_id = ?
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (thread_id, limit, offset),
                ).fetchall()
        return [
            {
                "run_id": row["run_id"],
                "thread_id": row["thread_id"],
                "assistant_id": row["assistant_id"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "status": row["status"],
                "metadata": json_loads(row["metadata_json"], {}),
            }
            for row in rows
        ]

    def update_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
        response: Any | None = None,
        error: Any | None = None,
        checkpoint_id: str | None = None,
        cancel_requested: bool | None = None,
    ) -> None:
        row = self.get_run_row(run_id)
        if row is None:
            raise KeyError(run_id)
        updated_at = now_iso()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE runs
                    SET status = COALESCE(?, status),
                        updated_at = ?,
                        response_json = COALESCE(?, response_json),
                        error_json = COALESCE(?, error_json),
                        checkpoint_id = COALESCE(?, checkpoint_id),
                        cancel_requested = COALESCE(?, cancel_requested)
                    WHERE run_id = ?
                    """,
                    (
                        status,
                        updated_at,
                        json_dumps(response) if response is not None else None,
                        json_dumps(error) if error is not None else None,
                        checkpoint_id,
                        int(cancel_requested) if cancel_requested is not None else None,
                        run_id,
                    ),
                )

    def set_run_cancel_requested(self, run_id: str, cancel_requested: bool = True) -> None:
        self.update_run(run_id, cancel_requested=cancel_requested)

    def is_run_cancel_requested(self, run_id: str) -> bool:
        row = self.get_run_row(run_id)
        if row is None:
            raise KeyError(run_id)
        return bool(row["cancel_requested"])

    def append_run_event(self, run_id: str, event: str, data: Any) -> int:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT COALESCE(MAX(event_id), 0) + 1 AS next_id FROM run_events WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                event_id = int(row["next_id"])
                conn.execute(
                    """
                    INSERT INTO run_events (run_id, event_id, event, data_json, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (run_id, event_id, event, json_dumps(data), now_iso()),
                )
                return event_id

    def get_run_events_since(self, run_id: str, last_event_id: int | None = None) -> list[dict[str, Any]]:
        last_event_id = last_event_id or 0
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT * FROM run_events
                    WHERE run_id = ? AND event_id > ?
                    ORDER BY event_id ASC
                    """,
                    (run_id, last_event_id),
                ).fetchall()
        return [
            {
                "event_id": row["event_id"],
                "event": row["event"],
                "data": json_loads(row["data_json"], None),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def all_threads(self) -> list[dict[str, Any]]:
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute("SELECT * FROM threads").fetchall()
        return [self._row_to_thread(row) for row in rows]

    def count_threads(
        self,
        *,
        metadata: Any | None = None,
        values: Any | None = None,
        status: str | None = None,
    ) -> int:
        threads = self.all_threads()
        count = 0
        for thread in threads:
            if metadata is not None and not is_subset(metadata, thread.get("metadata")):
                continue
            if values is not None and not is_subset(values, thread.get("values")):
                continue
            if status is not None and thread.get("status") != status:
                continue
            count += 1
        return count

    def search_threads(
        self,
        *,
        metadata: Any | None = None,
        ids: list[str] | None = None,
        limit: int = 10,
        offset: int = 0,
        status: str | None = None,
        sort_by: str | None = None,
        sort_order: str | None = None,
        select: list[str] | None = None,
        values: Any | None = None,
    ) -> list[dict[str, Any]]:
        threads = self.all_threads()
        filtered: list[dict[str, Any]] = []
        for thread in threads:
            if ids is not None and thread["thread_id"] not in ids:
                continue
            if metadata is not None and not is_subset(metadata, thread.get("metadata")):
                continue
            if values is not None and not is_subset(values, thread.get("values")):
                continue
            if status is not None and thread.get("status") != status:
                continue
            filtered.append(thread)

        sort_key = sort_by or "updated_at"
        reverse = (sort_order or "desc").lower() != "asc"
        filtered.sort(key=lambda item: item.get(sort_key) or "", reverse=reverse)
        sliced = filtered[offset : offset + limit if limit is not None else None]

        if not select:
            return sliced

        selected: list[dict[str, Any]] = []
        for thread in sliced:
            entry = {"thread_id": thread["thread_id"]}
            for field in select:
                if field in thread:
                    entry[field] = thread[field]
                elif field == "context":
                    entry[field] = None
            selected.append(entry)
        return selected
