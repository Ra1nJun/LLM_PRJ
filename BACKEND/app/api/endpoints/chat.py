from fastapi import APIRouter, Depends, HTTPException
from BACKEND.app.schemas.chat import ChatRequest, ChatResponse

from LLM.Agent import run

router = APIRouter()

@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    result = await run(request.message)
    return ChatResponse(answer=result)
