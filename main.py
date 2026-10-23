from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent
IMG_SIZE = 128
THRESHOLD = 0.5  # sigmoid chiqishi shu qiymatdan katta bo'lsa — parazit yuqqan

app = FastAPI(title="Malaria Cell Classifier")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

model = tf.keras.models.load_model(BASE_DIR / "malaria_cnn_model.h5")


def prepare_image(uploaded_file) -> np.ndarray:
    """Yuklangan rasmni modelga mos (1, 128, 128, 3) massivga aylantiradi, qiymatlar [0, 1] oralig'ida."""
    image = Image.open(uploaded_file.file).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    return np.expand_dims(np.array(image) / 255.0, axis=0)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        img_array = prepare_image(file)
    except (UnidentifiedImageError, OSError):
        return templates.TemplateResponse(
            request, "index.html",
            {"error": "Fayl rasm emas yoki buzilgan. PNG yoki JPG rasm yuklang."},
            status_code=400,
        )

    prediction = float(model.predict(img_array, verbose=0)[0][0])
    parasitic = prediction > THRESHOLD
    confidence = prediction if parasitic else 1 - prediction

    return templates.TemplateResponse(request, "index.html", {
        "result": "🦠 Parasitic" if parasitic else "✅ Uninfected",
        "confidence": f"{confidence * 100:.2f}%",
        "filename": file.filename,
    })
