from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# to get a string like this run:
# openssl rand -hex 32
# SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7" # first
SECRET_KEY = "pc36LXfFo0FVbDcH09R1oQU5qZlj4adqRWUfIvgN6ReAbN8r31rb7VPrXGZ7dE3Q"

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60*24*365 #minutes



users_db = {
    "go2joy": {
        "username": "go2joy",
        "full_name": "Backend",
        "email": "anh.tuan@go2joy.vn",
        "hashed_password": '$2b$12$1QTFlCgEY0DLn0SvBjc.kOqKc2vpA4vqQr/RxjrzEAdSxe.pnHIPC',
        "disabled": False,
    },
    "quan.nguyen": {
        "username": "quan.nguyen",
        "full_name": "Quan Nguyen - MKT",
        "email": "quan.nguyen@go2joy.vn",
        "hashed_password": '$2b$12$1QTFlCgEY0DLn0SvBjc.kOqKc2vpA4vqQr/RxjrzEAdSxe.pnHIPC',
        "disabled": False,
    }
}


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None


class UserInDB(User):
    hashed_password: str


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)


def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user



if __name__=="__main__":
    password = "P@ssword"
    h = get_password_hash(password)
    print(h)
    data = {"go2joy": {
        "username": "go2joy",
        "full_name": "Backend",
        "email": "anh.tuan@go2joy.vn",
        "hashed_password": '$2b$12$Ifupm3nrp552LHll4E.U5e.xPdzlJYhBS3JdYQdfw/F3.E/TycIj6',
        "disabled": False,
    }
    }
    token = create_access_token(data)
    print(token)