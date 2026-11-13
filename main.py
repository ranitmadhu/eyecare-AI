"""
EyeCare AI - FastAPI Entry Point
High-Accuracy Bilateral Retinal Screening System
"""

import uvicorn
from app import app

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)
