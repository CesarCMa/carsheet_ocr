# Carsheet OCR Backend

This is the backend service for the Carsheet OCR application, which processes images using OCR to extract relevant information.

## Technology Stack

- Python 3.9+
- FastAPI - Web framework
- Uvicorn - ASGI server
- Poetry - Dependency management

## Setup

1. Make sure you have Poetry installed:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies:
```bash
poetry install
```

3. Activate the virtual environment:
```bash
poetry shell
```

## Running the Application

To run the development server:

```bash
cd src
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

API Documentation will be available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Project Structure

```
backend/
├── src/
│   └── app/
│       ├── api/        # API endpoints
│       ├── core/       # Core functionality, config
│       ├── models/     # Data models
│       ├── services/   # Business logic
│       └── main.py     # Application entry point
├── tests/             # Test files
├── pyproject.toml     # Poetry dependency file
└── README.md         # This file
``` 