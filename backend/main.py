#author: anhlbt

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from .config import Config
from .app import create_app



from .app.models import User, Log

app = create_app(Config)

