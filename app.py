from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from io import BytesIO
from datetime import datetime
import os

app = Flask(__name__)
app.secret_key = "eyecare-ai-secret-key-change-this-later"
users = {}
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/login", methods=["GET", "POST"])
def login():
    if "user" in session:
        return redirect(url_for("home"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not email or not password:
            return render_template("login.html", error="Please enter your email and password.")
        if email not in users:
            return render_template("login.html", error="Account not found. Please create an account first.")
        if not check_password_hash(users[email]["password"], password):
            return render_template("login.html", error="Incorrect password. Please try again.")
        session["user"] = email
        session["name"] = users[email]["name"]
        return redirect(url_for("home"))
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if "user" in session:
        return redirect(url_for("home"))
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        if not name or not email or not password or not confirm_password:
            return render_template("register.html", error="Please complete all fields.")
        if len(password) < 6:
            return render_template("register.html", error="Password must contain at least 6 characters.")
        if password != confirm_password:
            return render_template("register.html", error="Passwords do not match.")
        if email in users:
            return render_template("register.html", error="An account with this email already exists.")
        users[email] = {"name": name, "password": generate_password_hash(password)}
        return redirect(url_for("login", registered="true"))
    return render_template("register.html")

@app.route("/")
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html", name=session.get("name"))

@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return jsonify({"error": "Please log in first."}), 401
    if "image" not in request.files:
        return jsonify({"error": "No image was uploaded."}), 400
    image = request.files["image"]
    if not image.filename:
        return jsonify({"error": "Please select an image."}), 400
    if not allowed_file(image.filename):
        return jsonify({"error": "Only JPG, JPEG and PNG images are allowed."}), 400
    filename = secure_filename(image.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    image.save(save_path)
    probabilities = {
        "Normal Retina": 8,
        "Cataract": 12,
        "Glaucoma": 18,
        "Diabetic Retinopathy": 62
    }
    label = max(probabilities, key=probabilities.get)
    return jsonify({
        "success": True,
        "mode": "demo",
        "label": label,
        "confidence": probabilities[label],
        "probabilities": probabilities,
        "message": "Demo result only. A trained ODIR-5K model will replace this prediction later."
    })

@app.route("/generate_report", methods=["POST"])
def generate_report():
    if "user" not in session:
        return jsonify({"error": "Please log in first."}), 401
    data = request.get_json(silent=True) or {}
    label = data.get("label", "No prediction")
    confidence = data.get("confidence", 0)
    probabilities = data.get("probabilities", {})
    filename = data.get("filename", "retinal_image")

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=45, leftMargin=45, topMargin=45, bottomMargin=45)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("EyeCare AI - Eye Screening Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Patient/User: {session.get('name', 'User')}", styles["Normal"]))
    story.append(Paragraph(f"Date: {datetime.now().strftime('%d %B %Y, %I:%M %p')}", styles["Normal"]))
    story.append(Paragraph(f"Image: {filename}", styles["Normal"]))
    story.append(Spacer(1, 18))
    story.append(Paragraph("Prediction Result", styles["Heading2"]))
    story.append(Paragraph(f"<b>Result:</b> {label}", styles["Normal"]))
    story.append(Paragraph(f"<b>Confidence:</b> {confidence}%", styles["Normal"]))
    story.append(Spacer(1, 12))
    rows = [["Condition", "Probability"]]
    for condition in ["Normal Retina", "Cataract", "Glaucoma", "Diabetic Retinopathy"]:
        rows.append([condition, f"{probabilities.get(condition, 0)}%"])
    table = Table(rows, colWidths=[300, 120])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eefc")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(table)
    story.append(Spacer(1, 20))
    story.append(Paragraph("Mode: DEMO - no trained ODIR-5K model is connected yet.", styles["Normal"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Disclaimer: This report is for demonstration/screening purposes only and is not a medical diagnosis. Please consult a qualified ophthalmologist for clinical evaluation.", styles["Normal"]))
    doc.build(story)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="EyeCare_AI_Report.pdf", mimetype="application/pdf")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({"error": "Image is too large. Maximum size is 10 MB."}), 413

if __name__ == "__main__":
    app.run(debug=True)
