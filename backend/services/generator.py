from __future__ import annotations
import os
import time
import random
from io import BytesIO
from pathlib import Path
from google import genai
from PIL import Image, ImageDraw

# Global variable to keep the model in memory
pipe = None

# A standard set of things we DON'T want in our art
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, extra limbs, grainy, text, watermark, "
    "signature, out of frame, deformed, disfigured, bad anatomy"
)

def get_gemini_client():
    api_key = os.getenv("GENAI_API_KEY")
    return genai.Client(api_key=api_key)

def refine_prompt_with_gemini(prompt: str, style: str, mood: str, palette: str) -> str:
    client = get_gemini_client()
    instruction = (
        "Expand this into a detailed Stable Diffusion v1.5 prompt. "
        "Focus on lighting, textures, and artistic composition. "
        "Keep it under 60 words. Output ONLY the final prompt text.\n\n"
        f"Subject: {prompt}, Style: {style}, Mood: {mood}, Colors: {palette}"
    )

    # List of models to try in order of preference
    model_candidates = ["gemini-1.5-flash", "gemini-1.5-flash-8b"]

    for model_name in model_candidates:
        for attempt in range(2):  # Try each model twice if quota hit
            try:
                response = client.models.generate_content(model=model_name, contents=instruction)
                return response.text.strip(), "ai"
            except Exception as e:
                error_str = str(e)
                if "429" in error_str:
                    wait_time = (attempt + 1) * 5
                    print(f"Quota hit for {model_name}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                elif "404" in error_str:
                    print(f"Model {model_name} not found, trying next candidate...")
                    break  # Break inner loop to try next model
                else:
                    print(f"Gemini error: {e}")
                    break
                    
    # Ultimate fallback if all API calls fail — use strong artistic descriptors
    style_modifiers = {
        "watercolor": "watercolor painting on paper, soft wet-on-wet brushstrokes, paint bleeding and blooming, visible paper grain, loose washes, NOT photorealistic, NOT 3D render",
        "cyberpunk": "cyberpunk digital art, neon-lit rain-soaked streets, glowing holographic signs, dark dystopian atmosphere, blade runner aesthetic",
        "charcoal-sketch": "charcoal sketch on paper, hand drawn, smudged charcoal marks, rough textured paper, high contrast black and white, NOT photorealistic",
        "pop-art": "pop art illustration, bold black outlines, flat halftone Ben-Day dots, Roy Lichtenstein style, primary colors, NOT photorealistic",
        "renaissance": "classical oil painting on canvas, visible brushstrokes, impasto technique, painted by Rembrandt, museum artwork, warm glazing layers, chiaroscuro lighting, NOT photorealistic, NOT photograph",
        "pixel-art": "pixel art, 16-bit, retro game style, pixelated, sprite art",
        "uquiyo-e": "ukiyo-e woodblock print, Japanese art, flat colors, bold outlines, Hokusai style",
        "surrealism": "surrealist painting, dreamlike, Salvador Dali style, impossible scene, melting forms",
    }
    mood_modifiers = {
        "vibrant": "vibrant colors, bright, vivid, energetic",
        "dark": "dark, moody, shadowy, dramatic lighting",
        "dreamy": "dreamy, soft focus, ethereal glow, pastel hues, misty",
        "ethereal": "ethereal, otherworldly, glowing, translucent",
        "cinematic": "cinematic, dramatic lighting, film still, wide angle",
    }
    style_text = style_modifiers.get(style, f"{style} art style") if style else "digital art"
    mood_text = mood_modifiers.get(mood, f"{mood} mood") if mood else ""
    return f"{prompt}, {style_text}, {mood_text}, {palette} color palette, highly detailed, masterpiece", "keyword"

def generate_with_diffusers(refined_prompt: str, output_dir: Path, filename: str, steps: int = 25, guidance_scale: float = 7.5, seed: int = -1) -> str:
    import torch
    from diffusers import StableDiffusionPipeline

    global pipe

    if pipe is None:
        print("--- Loading Stable Diffusion: Total Precision Mac Mode ---")
        model_id = "runwayml/stable-diffusion-v1-5"
        token = os.getenv("HF_TOKEN")

        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            variant="fp16",
            use_safetensors=True,
            token=token
        )

        pipe.safety_checker = None
        pipe.requires_safety_checker = False
        pipe = pipe.to("mps")
        pipe.enable_attention_slicing()

    print(f"--- Generating Image: {refined_prompt[:60]}... ---")

    try:
        actual_seed = seed if seed >= 0 else random.randint(0, 1000000)
        generator = torch.Generator("mps").manual_seed(actual_seed)

        image = pipe(
            prompt=refined_prompt,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            num_inference_steps=steps,
            generator=generator,
            guidance_scale=guidance_scale,
            height=512,
            width=512
        ).images[0]
        
        path = output_dir / filename
        image.save(path)
        return filename

    except Exception as e:
        print(f"Critical Generation Error: {e}")
        return generate_placeholder(refined_prompt, output_dir, filename)

def generate_with_imagen(refined_prompt: str, output_dir: Path, filename: str, seed: int = -1) -> str:
    import requests
    from urllib.parse import quote

    encoded = quote(refined_prompt)
    seed_param = f"&seed={seed}" if seed >= 0 else ""
    url = f"https://image.pollinations.ai/prompt/{encoded}?width=512&height=512&model=flux&nologo=true{seed_param}"

    response = requests.get(url, timeout=120)
    response.raise_for_status()

    path = output_dir / filename
    path.write_bytes(response.content)
    return filename

def generate_placeholder(refined_prompt: str, output_dir: Path, filename: str) -> str:
    img = Image.new("RGB", (512, 512), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)
    text = f"Prompt: {refined_prompt[:50]}...\n(Generation Error)"
    draw.multiline_text((40, 40), text, fill=(20, 20, 20))
    path = output_dir / filename
    img.save(path)
    return filename