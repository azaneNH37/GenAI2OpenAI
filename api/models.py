import logging

from flask import Blueprint, current_app, jsonify, g

from auth.cas_login import LoginError
from auth.token_manager import TokenExpiredError
from config import model_registry
from model_config.registry import MODEL_SPECS

logger = logging.getLogger(__name__)

models_bp = Blueprint('models', __name__)


def _fetch_models_with_renew(token: str):
    try:
        return model_registry.get_models(token), None
    except TokenExpiredError as e:
        first_error = e

    config = current_app.config["APP_CONFIG"]
    if not getattr(config, "fallback_renew", True):
        return None, first_error

    try:
        new_token = config.token_manager.force_refresh()
    except (LoginError, TokenExpiredError) as e:
        logger.warning("Force refresh failed during /v1/models fallback renew: %s", e)
        return None, e

    if not new_token:
        return None, first_error

    logger.info("Retrying model list after token renew")
    try:
        return model_registry.get_models(new_token), None
    except TokenExpiredError as e:
        return None, e


@models_bp.route('/v1/models', methods=['GET'])
def list_models():
    token = g.get("token", "")
    models_map, err = _fetch_models_with_renew(token)
    if err is not None:
        raise err

    combined = dict(models_map or {})
    for spec in MODEL_SPECS.values():
        combined.setdefault(spec.public_id, spec)

    config = current_app.config["APP_CONFIG"]
    for public_id, genai_id in getattr(config, "model_mapping", {}).items():
        combined.setdefault(public_id, _MappedModel(public_id, genai_id))

    models = []
    for model_id, info in combined.items():
        models.append({
            "id": model_id,
            "object": "model",
            "owned_by": info.root_ai_type,
            "permission": []
        })
    return jsonify({"object": "list", "data": models})


class _MappedModel:
    def __init__(self, public_id: str, genai_id: str):
        self.id = public_id
        self.name = public_id
        self.root_ai_type = "mapped"
        self.max_tokens = None
        self.description = f"mapped to {genai_id}"
