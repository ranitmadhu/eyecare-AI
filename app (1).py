"""
EyeCare AI — FastAPI backend (equivalent to the Flask version)

Serves the same /predict endpoint, same JSON response shape, same mock
model — just built on FastAPI instead of Flask. FastAPI gives you
automatic interactive docs at /docs, and built-in request validation.

Run with:
    uvicorn app:app --reload --port 5000

Then visit http://127.0.0.1:5000/docs to test it in the browser.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import random

app = FastAPI(title="EyeCare AI backend")

# allows the frontend (served from a different origin) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASSES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
MAX_FILE_SIZE_MB = 10


@app.get("/health")
def health():
    """Simple health check so you can confirm the server is up."""
    return {"status": "ok"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if not image.filename:
        raise HTTPException(status_code=400, detail="Empty filename.")

    file_bytes = await image.read()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=400, detail=f"File too large ({size_mb:.1f}MB). Max is {MAX_FILE_SIZE_MB}MB.")

    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read file as an image.")

    probabilities = run_inference(img)

    predicted_index = max(range(len(probabilities)), key=lambda i: probabilities[i])
    predicted_label = CLASSES[predicted_index]
    confidence = probabilities[predicted_index]

    return {
        "label": predicted_label,
        "confidence": round(confidence * 100, 1),
        "probabilities": {
            cls: round(prob * 100, 1) for cls, prob in zip(CLASSES, probabilities)
        }
    }


def run_inference(image: Image.Image):
    """
    Returns a list of 5 probabilities (summing to 1) in the order of CLASSES.

    --- REPLACE THIS FUNCTION WITH YOUR REAL MODEL ---
    Same idea as the Flask version — load the model once at module level,
    then run inference here. Example for a trained PyTorch model:

        import torch
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze().tolist()
        return probs
    """
    # --- MOCK LOGIC (remove once the real model is wired in) ---
    weights = [random.random() for _ in CLASSES]
    total = sum(weights)
    probabilities = [w / total for w in weights]
    return probabilities


# --- Uncomment and adapt this block once your trained model file exists ---
#
# import torch
# from torchvision import models
#
# model = models.efficientnet_b0(weights=None)
# model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(CLASSES))
# model.load_state_dict(torch.load("model_weights.pth", map_location="cpu"))
# model.eval()
