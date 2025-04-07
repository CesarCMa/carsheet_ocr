from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Carsheet OCR API",
    description="API for processing carsheet images using OCR",
    version="0.1.0"
)

# Configure CORS
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
    # For now, return a dummy response
    return JSONResponse(
        content={
            "status": "success",
            "message": "Image received successfully",
            "prediction": {
                "make": "Toyota",
                "model": "Camry",
                "year": 2020,
                "confidence": 0.95
            }
        }
    ) 