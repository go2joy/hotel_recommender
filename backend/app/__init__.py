
from logging.handlers import RotatingFileHandler

from redis import Redis
from flask import Flask, request
# from app.api import bp as api_bp
from .extensions import db, migrate
# from app.api import bp as api_bp

def create_app(config_class=None):
    '''Factory Pattern: Create Flask app.'''
    app = Flask(__name__)

    app.config.from_object(config_class)
    # Init Flask-SQLAlchemy
    db.init_app(app)
    # Init Flask-Migrate
    migrate.init_app(app, db)
    # app.register_blueprint(api_bp, url_prefix='/api')
    return app


