from fastapi import APIRouter, Depends, HTTPException
from BACKEND.app.schemas.user import UserCreate, UserResponse
from sqlalchemy.orm import Session
from BACKEND.app.db.session import get_db
from BACKEND.app.crud import user as crud_user

router = APIRouter()

@router.post("")
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    crud_user.create_user(db, user)
    return {"message": "Registered successfully"}