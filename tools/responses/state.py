_response_store: dict[str, dict] = {}
_history_store: dict[str, list[dict]] = {}


def store_response(response_id: str, payload: dict) -> None:
    _response_store[response_id] = payload


def get_response(response_id: str) -> dict | None:
    return _response_store.get(response_id)


def mark_cancelled(response_id: str) -> dict | None:
    payload = _response_store.get(response_id)
    if payload is None:
        return None
    payload = dict(payload)
    payload["status"] = "cancelled"
    _response_store[response_id] = payload
    return payload


def store_history(response_id: str, messages: list[dict]) -> None:
    _history_store[response_id] = list(messages)


def load_history(response_id: str | None) -> list[dict]:
    if not response_id:
        return []
    return list(_history_store.get(response_id, []))
