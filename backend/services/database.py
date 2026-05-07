from sqlmodel import Field, SQLModel, create_engine, Session
from sqlalchemy import text
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
    rating: int | None = Field(default=None)
    creator: str | None = Field(default=None)
    seed: int | None = Field(default=None)
    steps: int | None = Field(default=None)
    guidance_scale: float | None = Field(default=None)

sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"
# check_same_thread=False is required for SQLite to work with FastAPI's threads
engine = create_engine(sqlite_url, connect_args={"check_same_thread": False})

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)
    # Migrate existing DB: add rating column if not present
    with engine.connect() as conn:
        for col, typedef in [
            ("rating", "INTEGER"),
            ("creator", "TEXT"),
            ("seed", "INTEGER"),
            ("steps", "INTEGER"),
            ("guidance_scale", "REAL"),
        ]:
            try:
                conn.execute(text(f"ALTER TABLE imagemetadata ADD COLUMN {col} {typedef}"))
                conn.commit()
            except Exception:
                pass  # Column already exists