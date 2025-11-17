from sqlalchemy.orm import Session
from BACKEND.app.models.user import User
from BACKEND.app.schemas import user as UserSchema
from BACKEND.app.core import security

def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()

def create_user(db: Session, user: UserSchema.UserCreate):
    hashed_pw = security.hash_password(user.password)
    db_user = User(username=user.username, email=user.email, password=hashed_pw)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user
