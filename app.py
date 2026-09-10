from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash


app = Flask(__name__)

app.secret_key = "eyecare-ai-secret-key-change-this-later"


# Temporary users

users = {}


# =========================
# LOGIN PAGE
# =========================

@app.route("/login", methods=["GET", "POST"])
def login():

    
    if "user" in session:
        return redirect(url_for("home"))

    if request.method == "POST":

        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        # Empty field check
        if not email or not password:
            return render_template(
                "login.html",
                error="Please enter your email and password."
            )

        # User check
        if email not in users:
            return render_template(
                "login.html",
                error="Account not found. Please create an account first."
            )

        # Password check
        if not check_password_hash(users[email]["password"], password):
            return render_template(
                "login.html",
                error="Incorrect password. Please try again."
            )

        # Login successful
        session["user"] = email
        session["name"] = users[email]["name"]

        return redirect(url_for("home"))

    return render_template("login.html")


# =========================
# REGISTER PAGE
# =========================

@app.route("/register", methods=["GET", "POST"])
def register():

    # Already logged in
    if "user" in session:
        return redirect(url_for("home"))

    if request.method == "POST":

        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        # Empty field check
        if not name or not email or not password or not confirm_password:

            return render_template(
                "register.html",
                error="Please complete all fields."
            )

        # Password length
        if len(password) < 6:

            return render_template(
                "register.html",
                error="Password must contain at least 6 characters."
            )

        # Password match
        if password != confirm_password:

            return render_template(
                "register.html",
                error="Passwords do not match."
            )

        # Existing account
        if email in users:

            return render_template(
                "register.html",
                error="An account with this email already exists."
            )

        # Save user
        users[email] = {
            "name": name,
            "password": generate_password_hash(password)
        }

        return redirect(url_for("login", registered="true"))

    return render_template("register.html")


# =========================
# HOME PAGE
# =========================

@app.route("/")
def home():

    # Login না করলে login page
    if "user" not in session:
        return redirect(url_for("login"))

    return render_template(
        "index.html",
        name=session.get("name")
    )


# =========================
# LOGOUT
# =========================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("login"))


# =========================
# RUN SERVER
# =========================

if __name__ == "__main__":
    app.run(debug=True)