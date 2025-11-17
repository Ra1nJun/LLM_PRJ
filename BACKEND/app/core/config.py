from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ENV: str
    MariaDB_USER: str
    MariaDB_PW: str
    MariaDB_HOST: str
    MariaDB_NAME: str
    
    SECRET_KEY: str
    ALGORITHM: str
    TOKEN_EXPIRE_TIME: int

    @property
    def DATABASE_URL(self):
        return f"mysql+pymysql://{self.MariaDB_USER}:{self.MariaDB_PW}@{self.MariaDB_HOST}/{self.MariaDB_NAME}"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()