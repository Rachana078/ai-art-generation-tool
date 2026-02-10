from sqlmodel import Field, SQLModel, create_engine, Session
from datetime import datetime

class ImageMetadata(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    filename: str
    original_prompt: str
    refined_prompt: str
    style: str | None
    mood: str | None
    palette: str | None
    timestamp: datetime = Field(default_factory=datetime.now)

sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"
# check_same_thread=False is required for SQLite to work with FastAPI's threads
engine = create_engine(sqlite_url, connect_args={"check_same_thread": False})

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)