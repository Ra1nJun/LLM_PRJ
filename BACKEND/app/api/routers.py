from fastapi import APIRouter
from BACKEND.app.api.endpoints import chat, users, auth

api_router=APIRouter()
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])