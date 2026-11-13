# EyeCare AI - FastAPI Bilateral Retinal Screening System

High-accuracy bilateral retinal screening system migrated from Flask to **FastAPI** with asynchronous request handling, dual-eye ML pipeline support, Grad-CAM attention visualizer, SQLite persistent clinical history, and automated ReportLab PDF diagnostic reporting.

---

## 🚀 Quick Start

### 1. Using the Pre-configured Virtual Environment
You can run the application directly using your existing virtual environment (`fast api/myvenv`):

```powershell
# Navigate to the project directory
cd "C:\Users\Ranit Madhu\Desktop\New folder\eyecare-AI"

# Option A: Run directly with Python (starts Uvicorn on http://127.0.0.1:5000)
& "C:\Users\Ranit Madhu\Desktop\fast api\myvenv\Scripts\python.exe" app.py

# Option B: Run via Python's module runner (avoids Windows Application Control .exe blocks)
& "C:\Users\Ranit Madhu\Desktop\fast api\myvenv\Scripts\python.exe" -m uvicorn main:app --reload --port 5000
```

Open your browser at: **[http://localhost:5000](http://localhost:5000)**

---

## 🛠️ Key Migration Highlights (Flask to FastAPI)

1. **High-Performance ASGI Engine**:
   - Replaced WSGI Flask runtime with FastAPI + Uvicorn ASGI server.
   - Faster asynchronous file streaming and concurrent inference handling.

2. **Template & Static File Compatibility**:
   - Preserved all existing Jinja2 templates (`index.html`, `analysis.html`, `history.html`, `login.html`, `register.html`).
   - Seamless compatibility for Flask-style `url_for('static', filename='...')` and `request.args.get(...)` without needing any edits to HTML templates.

3. **Endpoints & Features**:
   - **`/`**: Home landing dashboard.
   - **`/login` & `/register`**: SQLite-backed session authentication.
   - **`/analysis`**: Bilateral dual-eye screening workspace.
   - **`/predict_bilateral`**: Asynchronous multipart upload for Left and Right fundus images with simultaneous disease grading and Grad-CAM generation.
   - **`/history` & `/api/history`**: Patient screening history logs stored in `eyecare.db`.
   - **`/generate_bilateral_report`**: Instant A4 PDF diagnostic report generation.
   - **`/docs`**: Interactive OpenAPI Swagger documentation available at `/docs`.

4. **Safety & Fallback**:
   - Original Flask implementation safely preserved at [`app_flask_backup.py`](app_flask_backup.py).
   - Dynamic AI pipeline fallback ensures the app operates in built-in clinical triage mode even if TensorFlow isn't present in the environment.
