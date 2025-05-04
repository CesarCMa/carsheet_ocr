import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.app.services.image_detection import detect_image as detect_image_service

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)  # Recurse into tuples
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    else:
        return obj

app = FastAPI(
    title="Carsheet OCR API",
    description="API for processing carsheet images using OCR",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to Carsheet OCR API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    predictions_raw, descriptions = detect_image_service(img)
    predictions_serializable = convert_numpy_types(predictions_raw)

    return JSONResponse(
        content={
            "status": "success",
            "message": "Image processed successfully",
            "predictions": predictions_serializable,
            "code_descriptions": descriptions
        }
    ) 