from fastapi import APIRouter
from BACKEND.app.api.endpoints import chat
from BACKEND.app.api.endpoints import users

api_router=APIRouter()
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(users.router, prefix="/users", tags=["users"])