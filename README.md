# AI Art Studio

A full-stack AI art generation tool that turns text prompts into styled artwork. Describe your vision, pick a style, mood, and palette — the backend refines your prompt with Gemini AI and generates an image via Stable Diffusion or Pollinations.

<img width="1470" height="828" alt="image" src="https://github.com/user-attachments/assets/50a08d27-7946-46c4-9549-ed5c8471eb75" />

<img width="1470" height="832" alt="image" src="https://github.com/user-attachments/assets/5a5810ec-b049-428f-83be-f6a9be5fe19e" />

---

## Features

- **Prompt refinement** — Gemini AI rewrites your prompt into a detailed generation-ready description
- **9 art styles** — Watercolor, Cyberpunk, Charcoal, Pop Art, Classic Oil, Pixel Art, Ukiyo-e, Surrealism, Custom
- **Mood selection** — Vibrant, Ethereal, Melancholic, Dramatic, Serene, Chaotic, or custom
- **Color palettes** — Colorful, Warm, Cool, Monochrome, Neon, Earthy
- **Gallery** — All past generations stored locally with pagination
- **Star ratings** — Rate any generated image 1–5 stars
- **Dark / light theme** — Toggle in the header
- **Per-user tagging** — Enter a name on first visit; your creations are tagged with it
- **Download** — One-click download of any generated image

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React, Vite, Axios |
| Backend | FastAPI, Python |
| Database | SQLite via SQLModel |
| AI (prompt) | Google Gemini 1.5 Flash |
| AI (image) | Stable Diffusion v1.5 (diffusers) or Pollinations.ai |

---

## Project Structure

```
ai-art-generation-tool/
├── backend/
│   ├── main.py                  # FastAPI app, routes
│   ├── services/
│   │   ├── generator.py         # Prompt refinement + image generation
│   │   └── database.py          # SQLModel schema + migrations
│   ├── outputs/images/          # Generated images (local)
│   ├── database.db              # SQLite database (local)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Main UI
│   │   └── studio.css           # All styles
│   └── index.html
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- A [Google Gemini API key](https://aistudio.google.com/app/apikey)

### 1. Clone the repo

```bash
git clone https://github.com/your-username/ai-art-generation-tool.git
cd ai-art-generation-tool
```

### 2. Backend setup

```bash
cd backend
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the `backend/` directory:

```env
GENAI_API_KEY=your_gemini_api_key_here

# Image provider: "placeholder" | "diffusers" | "imagen"
IMAGE_PROVIDER=imagen
```

Start the backend:

```bash
uvicorn main:app --reload
```

### 3. Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## Image Providers

| Value | Description |
|---|---|
| `placeholder` | Generates a grey placeholder image (no API needed, for testing) |
| `imagen` | Uses [Pollinations.ai](https://pollinations.ai) — free, no API key required |
| `diffusers` | Runs Stable Diffusion v1.5 locally via HuggingFace (requires `HF_TOKEN` and a capable GPU/MPS) |

Set `IMAGE_PROVIDER` in your `.env` to switch between them.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GENAI_API_KEY` | Yes | Google Gemini API key for prompt refinement |
| `IMAGE_PROVIDER` | No | `placeholder` / `imagen` / `diffusers` (default: `placeholder`) |
| `HF_TOKEN` | Only for `diffusers` | HuggingFace token to download Stable Diffusion |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/generate` | Generate an image from a prompt |
| `GET` | `/gallery` | Fetch past generations (paginated) |
| `PATCH` | `/api/generations/:id/rating` | Rate a generation |
| `DELETE` | `/api/generations/:id` | Delete a generation |
| `GET` | `/health` | Health check |
