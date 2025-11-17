from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordRequestForm
from BACKEND.app.db.session import get_db
from BACKEND.app.core.security import verify_password, create_access_token
from BACKEND.app import crud, models
from BACKEND.app.core.security import get_current_user

router = APIRouter()

@router.post("/login")
def login(response: Response, form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = crud.user.get_user_by_email(db, email=form_data.username)
    if not user or not verify_password(form_data.password, user.password):
        raise HTTPException(status_code=400, detail="Invalid id or password")

    access_token = create_access_token(data={"sub": user.email})
    response.set_cookie(key="access_token", value=access_token, httponly=True, max_age=60*60, samesite="strict", secure=True)
    return {"message": "Logged in successfully"}

@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(key="access_token")
    return {"message": "Logged out successfully"}

@router.get("/me")
async def token_validation(current_user=Depends(get_current_user)):
    return Response(status_code=204)