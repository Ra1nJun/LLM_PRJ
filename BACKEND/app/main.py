from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from BACKEND.app.api.routers import api_router

app = FastAPI(
    title="Haru Dang",
    description="Dog Care AI",
    version="0.0.1",
    docs_url="/docs",
    redoc_url="/redoc"
)

origins = ["https://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 허용할 출처 목록
    allow_credentials=True, # 쿠키/인증 헤더를 허용할지 여부
    allow_methods=["*"],    # 모든 HTTP 메서드 (GET, POST, PUT 등) 허용
    allow_headers=["*"],    # 모든 HTTP 헤더 허용
)

app.include_router(api_router)