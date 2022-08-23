# author: anhlbt
from datetime import datetime, timedelta
from .extensions import db




class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    email = db.Column(db.String(120), index=True, unique=True)
    password_hash = db.Column(db.String(128))
    name = db.Column(db.String(64))
    location = db.Column(db.String(64))
    member_since = db.Column(db.DateTime(), default=datetime.now)

class Log(db.Model):
    __tablename__ = 'logs'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20))
    ip = db.Column(db.String(50))
    file_name = db.Column(db.String(200))
    timestamp = db.Column(db.DateTime, index=True, default=datetime.now)
    # ProposalNo	AgentID	ConfirmDate	URL	PO	LI	AL	
    proposal_no= db.Column(db.String(20))
    agent_id= db.Column(db.String(20))
    confirm_date = db.Column(db.DateTime)
    is_po_null = db.Column(db.Boolean())
    is_li_null = db.Column(db.Boolean())
    is_al1_null = db.Column(db.Boolean())
    is_al2_null = db.Column(db.Boolean())
    is_al3_null = db.Column(db.Boolean())
    is_agent_null = db.Column(db.Boolean())
