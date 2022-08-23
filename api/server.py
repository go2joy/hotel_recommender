from email import message
from os.path import dirname, join, realpath, isfile
import sys
from pandas import DataFrame
C_DIR = dirname(realpath(__file__))
P_DIR = dirname(C_DIR)
sys.path.insert(0, P_DIR)
import uvicorn
from fastapi import FastAPI, File, UploadFile, Depends
from starlette.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from celery import states
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse, JSONResponse
from apscheduler.schedulers.background import BackgroundScheduler
# from backend.app.extensions import db
# from backend.app.models import Log
# from backend.main import app as flask_app
from typing import List, Optional, Any
# from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from PIL import Image, ImageOps
from io import BytesIO
import base64
from starlette.status import (
    HTTP_200_OK,
    HTTP_201_CREATED
)
from utils.util import exception_logger, format_checking, logger
import uuid
import numpy as np
from worker.app_worker import celery_app
from api.auth import *
from fastapi_utils.tasks import repeat_every
import datetime
from db.database import MyDatabase
from go2joy import HotelRecommender

app_desc = """<h2>GO2JOY</h2>
"""
class Hotel(BaseModel):
    booking_type: int = 1
    hotel_sn: int = 1141
    radius: float = None
    nearest_hotel_num: int = 10
    alpha: float = .5
    threshold: float = 0
    ip: str = "118.69.104.242"

class TaskResult(BaseModel):
    id: str
    message: str
    error: Optional[str] = None
    result: Optional[Any] = None
    
TASKS = {
    "get_weight_matrix": "worker.tasks.weight_matrix",
}


app = FastAPI(title='Hotel recommendation system API', description=app_desc)
origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
UPLOAD_FOLDER = 'uploads'
dataframe = None
recommender = HotelRecommender()


def init_data():
    global recommender
    recommender = HotelRecommender()
    logger.info(f'get new data at: {recommender.datetime}')



@app.on_event('startup')
def get_data():
    scheduler = BackgroundScheduler()
    scheduler.add_job(init_data, 'cron', day_of_week=recommender.config['cron']['day_of_week'], \
        hour=recommender.config['cron']['hour'], \
            minute=recommender.config['cron']['minute']) #second='*/5'
    scheduler.start()

@app.get('/info', tags=['Utility'])
def info():
    """
    """
    try:
        about = "__anhlbt__"
        return JSONResponse(status_code=200, content={"status":True,"result": {"author": about}})
    except:
        return JSONResponse(status_code=400, content={"status":False})


@app.post("/token", response_model=Token, tags=['Token'])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    try:
        user = authenticate_user(users_db, form_data.username, form_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        # print("access_token_expires: ", access_token_expires)
        access_token = create_access_token(
            data={"sub": user.username}, expires_delta=access_token_expires
        )
        # return {"status":True, "result":{"access_token": access_token, "token_type": "bearer"}}
        return {"access_token": access_token, "token_type": "bearer"}
    except:
        return JSONResponse(status_code=400, content={"status":False})   

@app.get("/users/me/", response_model=User, tags=['Token'])
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user


@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")


def send_result(task):
    while True:
        result = celery_app.AsyncResult(task.id)
        if result.state in states.READY_STATES:
            break
    output = TaskResult(
        id=task.id,
        message=result.state,
        error=str(result.info) if result.failed() else None,
        result=result.get() if result.state == states.SUCCESS else None
    )
    return output

@app.get("/task/{task_id}")
def get_task_result(task_id: str):
    result = celery_app.AsyncResult(task_id)
    output = TaskResult(
        id=task_id,
        message=result.state,
        error=str(result.info) if result.failed() else None,
        result=result.get() if result.state == states.SUCCESS else None
    )
    return JSONResponse(
        status_code=200,
        content={"status":True, "data": output.dict()}
        # content=output.dict()
    )



@app.post("/hotel_recommender", tags=['HPRS'])
async def get_price(hotel: Hotel, request: Request, current_user: User = Depends(get_current_active_user)):
    # ...
    try:
        global recommender
        res=None
        task_id = str(uuid.uuid4())
        # host = request.client.host
        # username = current_user.username
        host = hotel.ip #
        username = hotel.hotel_sn
        weight_matrix = recommender.get_weight_matrix(hotel.hotel_sn , hotel.radius, hotel.nearest_hotel_num,  hotel.alpha, hotel.threshold, visual = False)         
        res = weight_matrix.to_dict()
        return JSONResponse(status_code=200, content={"status":True, "data":res})   

    except Exception as ex:
        logger.exception(ex)
        return JSONResponse(status_code=400, content={"status":False, "data":{}})
    finally:
        try:
            result = {
            "username": username, "ip": host ,"hotel_sn": hotel, "task_id": task_id, "results": res,
            }
            logger.info(result)   
        except Exception as ex:
            logger.exception(ex)




if __name__ == "__main__":
    uvicorn.run("server:app", debug=True, host="0.0.0.0", port=5005, workers=4)

    

