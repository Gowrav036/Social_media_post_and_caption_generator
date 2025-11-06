import os
import time
import io
from typing import Optional
from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai
import gradio as gr
from dotenv import load_dotenv

# Load .env (if present) and then read from environment
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set. Please set it in your environment before running.")

genai.configure(api_key=API_KEY)

# Ensure the symbol exists even under odd import/reload scenarios
if 'gemini_model' not in globals():
    gemini_model = None

def _pick_model():
    # Prefer 1.5 family models known to support generateContent
    candidates = [
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.5-pro",
    ]
    last_err = None
    for name in candidates:
        try:
            m = genai.GenerativeModel(name)
            # quick ping
            _ = m.generate_content("ping")
            print(f"Using model: {name}")
            return m
        except Exception as e:
            last_err = e
            continue

    # Fallback: query available models and pick a supported one dynamically
    try:
        models = list(getattr(genai, "list_models")())
        # Model names may be returned as 'models/<name>'
        def normalize(n: str) -> str:
            return n.split("/", 1)[1] if n.startswith("models/") else n

        # Prefer 1.5 models that support generateContent
        for md in models:
            methods = getattr(md, "supported_generation_methods", []) or []
            if "generateContent" in methods:
                name = normalize(getattr(md, "name", ""))
                if name.startswith("gemini-1.5-"):
                    try:
                        m = genai.GenerativeModel(name)
                        _ = m.generate_content("ping")
                        print(f"Using model (fallback): {name}")
                        return m
                    except Exception as e:
                        last_err = e
                        continue

        # Final fallback: any model that supports generateContent
        for md in models:
            methods = getattr(md, "supported_generation_methods", []) or []
            if "generateContent" in methods:
                name = normalize(getattr(md, "name", ""))
                try:
                    m = genai.GenerativeModel(name)
                    _ = m.generate_content("ping")
                    print(f"Using model (final fallback): {name}")
                    return m
                except Exception as e:
                    last_err = e
                    continue
    except Exception as e:
        last_err = e

    raise RuntimeError(f"No available Gemini model from candidates or list_models. Last error: {last_err}")

try:
    gemini_model = _pick_model()
    print("Gemini model initialized successfully.")
except Exception as e:
    print(f"Error initializing Gemini: {e}")


def generate_social_media_post(keywords, tone, platform, hashtag_count):
    # Access and set via globals to avoid NameError under threaded contexts
    gm = globals().get('gemini_model', None)
    if gm is None:
        try:
            gm = _pick_model()
            globals()['gemini_model'] = gm
        except Exception as e:
            return f"Model initialization failed: {e}"
    # Basic validation
    if not keywords:
        return "Please enter some keywords."

    prompt = (
        f"Generate a {tone} social media caption for {platform} with keywords: {keywords}. "
        f"Include exactly {hashtag_count} hashtags and a few emojis. Format as Caption, Hashtags, Emojis."
    )

    # Server-side diagnostics (visible in terminal)
    try:
        print("Calling Gemini with:", {
            "tone": tone,
            "platform": platform,
            "hashtag_count": hashtag_count,
            "keywords_len": len(keywords or "")
        })
    except Exception:
        pass

    try:
        last_err = None
        for _ in range(2):
            try:
                response = gm.generate_content(prompt)
                # Handle different response shapes defensively
                if hasattr(response, "text") and response.text:
                    return response.text
                # Fallback: try to extract from candidates
                try:
                    cands = getattr(response, "candidates", []) or []
                    parts = []
                    for c in cands:
                        content = getattr(c, "content", None)
                        if content and hasattr(content, "parts"):
                            for p in content.parts:
                                if hasattr(p, "text") and p.text:
                                    parts.append(p.text)
                    if parts:
                        return "\n".join(parts)
                except Exception:
                    pass
                # If still nothing, return a friendly message
                return "The model returned an empty response. Please try different inputs or try again."
            except Exception as e:
                last_err = e
                time.sleep(2)
        return f"Failed to generate content after retries. Error: {last_err}"
    except Exception as e:
        return f"Unexpected error: {e}"


def _create_synthetic_image(text: str, size=(768, 768)) -> Image.Image:
    img = Image.new("RGB", size, (245, 245, 245))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    margin = 24
    y = margin
    max_w = size[0] - margin * 2
    for line in _wrap_text(draw, text, font, max_w):
        draw.text((margin, y), line, font=font, fill=(20, 20, 20))
        y += 28
        if y > size[1] - margin:
            break
    return img

def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_w: int):
    words = (text or "").split()
    line = []
    for w in words:
        test = " ".join(line + [w])
        if draw.textlength(test, font=font) <= max_w:
            line.append(w)
        else:
            if line:
                yield " ".join(line)
            line = [w]
    if line:
        yield " ".join(line)

def caption_from_image(image: Optional[Image.Image], description: str, tone: str, platform: str, hashtag_count: str) -> str:
    gm = globals().get('gemini_model', None)
    if gm is None:
        try:
            gm = _pick_model()
            globals()['gemini_model'] = gm
        except Exception as e:
            return f"Model initialization failed: {e}"
    if image is None:
        return "Please upload an image."
    prompt = (
        f"You are a social media assistant. Using the provided image and hints, write a {tone} caption for {platform}. "
        f"Include exactly {hashtag_count} hashtags and a few emojis. If a short description is given, incorporate it."
    )
    try:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_part = {"mime_type": "image/png", "data": buf.getvalue()}
        parts = [prompt]
        if description:
            parts.append(f"Description: {description}")
        parts.append(img_part)
        r = gm.generate_content(parts)
        if getattr(r, "text", None):
            return r.text
        return "No caption produced. Try another image."
    except Exception as e:
        return f"Failed to generate caption: {e}"

def image_from_caption(caption: str) -> Image.Image:
    if not caption:
        return _create_synthetic_image("No caption provided")
    try:
        return _create_synthetic_image(caption)
    except Exception:
        return _create_synthetic_image(caption)

def both_from_keywords(keywords: str, tone: str, platform: str, hashtag_count: str):
    cap = generate_social_media_post(keywords, tone, platform, hashtag_count)
    img = image_from_caption(cap)
    return cap, img

with gr.Blocks() as demo:
    gr.Markdown("## 🌐 Social Media Post & Image Generator")

    mode = gr.Radio([
        "Photo → Caption",
        "Caption → Photo",
        "Keywords → Both",
    ], value="Keywords → Both", label="Mode")

    with gr.Group() as kw_group:
        with gr.Row():
            keywords_input = gr.Textbox(label="Keywords", placeholder="e.g., coffee morning vibes")
            tone_input = gr.Dropdown(["Funny", "Professional", "Motivational", "Casual", "Inspirational"], value="Casual", label="Tone")
        with gr.Row():
            platform_input = gr.Dropdown(["Instagram", "LinkedIn", "Twitter"], value="Instagram", label="Platform")
            hashtag_input = gr.Dropdown(["3", "5", "10"], value="5", label="Number of Hashtags")
        out_caption_kw = gr.Textbox(label="Generated Caption, Hashtags & Emojis", lines=8)
        out_image_kw = gr.Image(label="Generated Image", type="pil")
        btn_kw = gr.Button("Generate Both")
        btn_kw.click(
            fn=both_from_keywords,
            inputs=[keywords_input, tone_input, platform_input, hashtag_input],
            outputs=[out_caption_kw, out_image_kw],
        )

    with gr.Group(visible=False) as cap_group:
        caption_in = gr.Textbox(label="Caption / Prompt", placeholder="Describe the desired image")
        out_image_cap = gr.Image(label="Generated Image", type="pil")
        btn_cap = gr.Button("Generate Photo")
        btn_cap.click(
            fn=image_from_caption,
            inputs=[caption_in],
            outputs=[out_image_cap],
        )

    with gr.Group(visible=False) as img_group:
        img_in = gr.Image(label="Upload Photo", type="pil")
        desc_in = gr.Textbox(label="Short Description (optional)")
        tone_in2 = gr.Dropdown(["Funny", "Professional", "Motivational", "Casual", "Inspirational"], value="Casual", label="Tone")
        platform_in2 = gr.Dropdown(["Instagram", "LinkedIn", "Twitter"], value="Instagram", label="Platform")
        hashtag_in2 = gr.Dropdown(["3", "5", "10"], value="5", label="Number of Hashtags")
        out_caption_img = gr.Textbox(label="Generated Caption", lines=8)
        btn_img = gr.Button("Generate Caption from Photo")
        btn_img.click(
            fn=caption_from_image,
            inputs=[img_in, desc_in, tone_in2, platform_in2, hashtag_in2],
            outputs=[out_caption_img],
        )

    def _switch(u_mode):
        return (
            gr.update(visible=(u_mode == "Keywords → Both")),
            gr.update(visible=(u_mode == "Caption → Photo")),
            gr.update(visible=(u_mode == "Photo → Caption")),
        )

    mode.change(_switch, inputs=[mode], outputs=[kw_group, cap_group, img_group])

if __name__ == "__main__":
    # Allow dynamic port selection via env; otherwise use 7871 to avoid conflicts
    port_env = os.getenv("GRADIO_SERVER_PORT")
    try:
        port_val = int(port_env) if (port_env and port_env.strip()) else 7871
    except Exception:
        port_val = 7871
    demo.launch(server_name="0.0.0.0", server_port=port_val, debug=True, show_error=True, share=True)
