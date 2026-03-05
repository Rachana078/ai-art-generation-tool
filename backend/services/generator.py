from __future__ import annotations
import os
import torch
import time
import random
from pathlib import Path
from google import genai
from PIL import Image, ImageDraw
from diffusers import StableDiffusionPipeline

# Global variable to keep the model in memory
pipe = None

# A standard set of things we DON'T want in our art
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, extra limbs, grainy, text, watermark, "
    "signature, out of frame, deformed, disfigured, bad anatomy"
)

def get_gemini_client():
    api_key = os.getenv("GENAI_API_KEY")
    return genai.Client(api_key=api_key, http_options={'api_version': 'v1'})

def refine_prompt_with_gemini(prompt: str, style: str, mood: str, palette: str) -> str:
    client = get_gemini_client()
    instruction = (
        "Expand this into a detailed Stable Diffusion v1.5 prompt. "
        "Focus on lighting, textures, and artistic composition. "
        "Keep it under 60 words. Output ONLY the final prompt text.\n\n"
        f"Subject: {prompt}, Style: {style}, Mood: {mood}, Colors: {palette}"
    )

    # List of models to try in order of preference
    model_candidates = ["gemini-1.5-flash", "gemini-2.0-flash"]
    
    for model_name in model_candidates:
        for attempt in range(2):  # Try each model twice if quota hit
            try:
                response = client.models.generate_content(model=model_name, contents=instruction)
                return response.text.strip()
            except Exception as e:
                error_str = str(e)
                if "429" in error_str:
                    wait_time = (attempt + 1) * 5
                    print(f"Quota hit for {model_name}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                elif "404" in error_str:
                    print(f"Model {model_name} not found, trying next candidate...")
                    break # Break inner loop to try next model
                else:
                    print(f"Gemini error: {e}")
                    break
                    
    # Ultimate fallback if all API calls fail
    return f"{prompt}, {style} style, {mood} mood, {palette} palette, highly detailed, 8k"

def generate_with_diffusers(refined_prompt: str, output_dir: Path, filename: str) -> str:
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
        generator = torch.Generator("mps").manual_seed(random.randint(0, 1000000))
        
        image = pipe(
            prompt=refined_prompt,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT, # Added for quality
            num_inference_steps=25, # Slightly more steps for better quality
            generator=generator,
            guidance_scale=7.5,     # Controls how strictly it follows the prompt
            height=512,
            width=512
        ).images[0]
        
        path = output_dir / filename
        image.save(path)
        return filename

    except Exception as e:
        print(f"Critical Generation Error: {e}")
        return generate_placeholder(refined_prompt, output_dir, filename)

def generate_placeholder(refined_prompt: str, output_dir: Path, filename: str) -> str:
    img = Image.new("RGB", (512, 512), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)
    text = f"Prompt: {refined_prompt[:50]}...\n(Generation Error)"
    draw.multiline_text((40, 40), text, fill=(20, 20, 20))
    path = output_dir / filename
    img.save(path)
    return filename