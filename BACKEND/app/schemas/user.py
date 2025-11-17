from pydantic import BaseModel, EmailStr, Field

# ...은 반드시 필드가 반드시 있어야 한다는 표현
class UserCreate(BaseModel):
    username: str = Field(..., min_length=1, max_length=20)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=60)
    confirm_password: str = Field(..., min_length=8, max_length=60)

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: EmailStr

    class Config:
        orm_mode = True
