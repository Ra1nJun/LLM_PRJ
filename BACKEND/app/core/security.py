from jose import JWTError, jwt
from datetime import datetime, timedelta
from passlib.hash import bcrypt
from BACKEND.app.core.config import settings
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from BACKEND.app.db.session import get_db

SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.ALGORITHM
TOKEN_EXPIRE_TIME = settings.TOKEN_EXPIRE_TIME
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def hash_password(password: str):
    return bcrypt.hash(password)

def verify_password(plain_pw, hashed_pw):
    return bcrypt.verify(plain_pw, hashed_pw)

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    now = datetime.now()
    to_encode.update({"iat": now})
    expire = now + (expires_delta or timedelta(minutes=TOKEN_EXPIRE_TIME))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")  # 또는 payload["email"] 등 클레임 위치에 따라
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    from BACKEND.app.crud.user import get_user_by_email
    user = get_user_by_email(db, email)
    if user is None:
        raise credentials_exception
    return user