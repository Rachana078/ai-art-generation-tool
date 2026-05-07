import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles

# 1. LOAD DOTENV FIRST
load_dotenv()

# 2. NOW IMPORT OTHER MODULES
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import Field as PydanticField
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlmodel import Session, select

# Import your local modules
from services.database import engine, create_db_and_tables, ImageMetadata  # Updated path
from services.generator import (
    generate_placeholder,
    refine_prompt_with_gemini,
    generate_with_diffusers,
    generate_with_imagen,
)

GENAI_PROVIDER = os.getenv("GENAI_PROVIDER", "gemini").lower()
IMAGE_PROVIDER = os.getenv("IMAGE_PROVIDER", "placeholder").lower()

app = FastAPI(title="AI Art Generation Tool API")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

@app.on_event("startup")
def on_startup():
    create_db_and_tables()

class GenerateRequest(BaseModel):
    prompt: str
    style: str | None = None
    mood: str | None = None
    palette: str | None = None
    creator: str | None = None

class RatingRequest(BaseModel):
    rating: int = PydanticField(..., ge=1, le=5)

# OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "images"
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR.parent), name="outputs")

# This ensures it looks for 'outputs' inside the same folder as main.py
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Update the mount to use the correct local path
app.mount("/outputs", StaticFiles(directory=BASE_DIR / "outputs"), name="outputs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "Backend is running ✅"}

@app.get("/images/{filename}")
def get_image(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)

@app.get("/gallery")
def gallery(limit: int = 20, offset: int = 0):
    limit = max(1, min(limit, 100))
    with Session(engine) as session:
        total = session.exec(select(ImageMetadata)).all()
        statement = (
            select(ImageMetadata)
            .order_by(ImageMetadata.timestamp.desc())
            .offset(offset)
            .limit(limit)
        )
        records = session.exec(statement).all()

    items = []
    for record in records:
        items.append({
            "id": record.id,
            "filename": record.filename,
            "image_path": f"/outputs/images/{record.filename}",
            "original_prompt": record.original_prompt,
            "refined_prompt": record.refined_prompt,
            "style": record.style,
            "mood": record.mood,
            "palette": record.palette,
            "timestamp": record.timestamp.isoformat(),
            "rating": record.rating,
            "creator": record.creator,
            "seed": record.seed,
            "steps": record.steps,
            "guidance_scale": record.guidance_scale,
        })
    return {"total": len(total), "count": len(items), "offset": offset, "items": items}


@app.delete("/api/generations/{generation_id}")
def delete_generation(generation_id: int):
    with Session(engine) as session:
        record = session.get(ImageMetadata, generation_id)
        if not record:
            raise HTTPException(status_code=404, detail="Generation not found")
        file_path = OUTPUT_DIR / record.filename
        if file_path.exists():
            file_path.unlink()
        session.delete(record)
        session.commit()
    return {"status": "deleted", "id": generation_id}

@app.patch("/api/generations/{generation_id}/rating")
def rate_generation(generation_id: int, payload: RatingRequest):
    with Session(engine) as session:
        record = session.get(ImageMetadata, generation_id)
        if not record:
            raise HTTPException(status_code=404, detail="Generation not found")
        record.rating = payload.rating
        session.add(record)
        session.commit()
        session.refresh(record)
    return {"id": record.id, "rating": record.rating}

# @app.post("/generate")
# async def generate_art(payload: GenerateRequest, http: Request, background_tasks: BackgroundTasks):
#     refined_prompt = refine_prompt_with_gemini(
#         payload.prompt, payload.style, payload.mood, payload.palette
#     )

#     filename = f"art_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"

#     def background_generation_task():
#         try:
#             if IMAGE_PROVIDER == "diffusers":
#                 generate_with_diffusers(refined_prompt, OUTPUT_DIR, filename)
#             else:
#                 generate_placeholder(payload.prompt, payload.style, payload.mood, payload.palette, OUTPUT_DIR, filename)
#         except Exception as e:
#             print(f"Generation failed: {e}")
#             generate_placeholder(payload.prompt, payload.style, payload.mood, payload.palette, OUTPUT_DIR, filename)

#         with Session(engine) as session:
#             meta = ImageMetadata(
#                 filename=filename,
#                 original_prompt=payload.prompt,
#                 refined_prompt=refined_prompt,
#                 style=payload.style,
#                 mood=payload.mood,
#                 palette=payload.palette
#             )
#             session.add(meta)
#             session.commit()

#     background_tasks.add_task(background_generation_task)

#     base_url = str(http.base_url).rstrip("/")
#     return {
#         "status": "processing",
#         "filename": filename,
#         "refined_prompt": refined_prompt,
#         "image_url": f"{base_url}/outputs/images/{filename}",
#         "mode": IMAGE_PROVIDER
#     }

@app.post("/generate")
async def generate_art(payload: GenerateRequest, http: Request):
    # 1. Refine the prompt
    refined_prompt, prompt_source = refine_prompt_with_gemini(
        payload.prompt, payload.style, payload.mood, payload.palette
    )

    filename = f"art_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"

    # 2. RUN THE GENERATION DIRECTLY (Remove background_tasks)
    # This forces the backend to wait until the image is saved to disk
    try:
        if IMAGE_PROVIDER == "diffusers":
            generate_with_diffusers(refined_prompt, OUTPUT_DIR, filename)
        elif IMAGE_PROVIDER == "placeholder":
            generate_placeholder(refined_prompt, OUTPUT_DIR, filename)
        else:
            generate_with_imagen(refined_prompt, OUTPUT_DIR, filename)
    except Exception as e:
        print(f"Generation failed: {e}")
        generate_placeholder(refined_prompt, OUTPUT_DIR, filename)

    # 3. Save to Database
    with Session(engine) as session:
        meta = ImageMetadata(
            filename=filename,
            original_prompt=payload.prompt,
            refined_prompt=refined_prompt,
            style=payload.style,
            mood=payload.mood,
            palette=payload.palette,
            creator=payload.creator,
        )
        session.add(meta)
        session.commit()

    # 4. Now that the file is 100% SAVED, return the URL
    base_url = str(http.base_url).rstrip("/")
    return {
        "status": "completed",
        "filename": filename,
        "refined_prompt": refined_prompt,
        "prompt_source": prompt_source,
        "image_url": f"{base_url}/outputs/images/{filename}",
        "mode": IMAGE_PROVIDER
    }