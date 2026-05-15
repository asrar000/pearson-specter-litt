from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from backend.config import settings

engine = create_engine(
    f"sqlite:///{settings.SQLITE_DB_PATH}",
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from backend import models  # noqa: F401 — registers models with Base
    Base.metadata.create_all(bind=engine)
