from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from BACKEND.app.core.config import settings
import pymysql
from BACKEND.app.db.base import Base
from BACKEND.app.models import user

if settings.ENV == "dev":
    conn = pymysql.connect(
        host=settings.MariaDB_HOST,
        user=settings.MariaDB_USER,
        password=settings.MariaDB_PW
    )
    with conn.cursor() as cursor:
        cursor.execute(f"DROP DATABASE IF EXISTS {settings.MariaDB_NAME};")
        cursor.execute(f"CREATE DATABASE {settings.MariaDB_NAME} CHARACTER SET utf8mb4;")
    conn.close()
    print(f"[DB INIT] '{settings.MariaDB_NAME}' 새로 생성 완료 ✅")

engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()