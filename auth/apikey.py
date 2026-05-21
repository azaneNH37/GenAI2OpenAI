from flask import current_app, request

from errors import openai_error
from compat.anthropic import anthropic_error


def register_auth(app):
    @app.before_request
    def check_api_key():
        config = current_app.config["APP_CONFIG"]
        if not config.api_key:
            return

        if request.path == '/health':
            return

        if not request.path.startswith('/v1/'):
            return

        if request.path.startswith('/v1/messages'):
            client_key = request.headers.get('x-api-key')
            auth_header = request.headers.get('Authorization', '')
            if not client_key and auth_header.startswith('Bearer '):
                client_key = auth_header[7:]
            if not client_key:
                return anthropic_error("Missing API key", "authentication_error", 401)
            if client_key != config.api_key:
                return anthropic_error("Invalid API key", "authentication_error", 401)
            return

        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return openai_error(
                "Missing Authorization header with Bearer token",
                error_type="invalid_request_error",
                code="invalid_api_key",
                status=401
            )

        token = auth_header[7:]
        if token != config.api_key:
            return openai_error(
                "Incorrect API key provided",
                error_type="invalid_request_error",
                code="invalid_api_key",
                status=401
            )
