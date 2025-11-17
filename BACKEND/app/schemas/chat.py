from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    user_email: Optional[str] = None
    message: str

class ChatResponse(BaseModel):
    user_email: Optional[str] = None
    answer: str