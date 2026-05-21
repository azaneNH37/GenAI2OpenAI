from api.chat import chat_bp
from api.anthropic import anthropic_bp
from api.models import models_bp
from api.health import health_bp
from api.responses import responses_bp


def register_routes(app):
    app.register_blueprint(chat_bp)
    app.register_blueprint(anthropic_bp)
    app.register_blueprint(models_bp)
    app.register_blueprint(health_bp)
    app.register_blueprint(responses_bp)
