"""Flask web application for AI-Based Image Colorization with user authentication."""

import os
import uuid
from functools import wraps

from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
    abort,
    session,
    flash,
)

import colorizer
import sketcher
import database

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_UPLOAD_SIZE = 5 * 1024 * 1024


def load_env():
    """Load simple KEY=VALUE pairs from a local .env file if present."""
    env_path = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())
        except OSError:
            pass


load_env()

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE
app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY", "dev-secret-change-me-in-production"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
database.init_db()


def allowed_file(filename):
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def safe_join(directory, filename):
    if not filename:
        return None
    filename = os.path.basename(filename)
    filename = secure_filename(filename)
    if not filename:
        return None
    full_path = os.path.abspath(os.path.join(directory, filename))
    if not full_path.startswith(os.path.abspath(directory) + os.sep):
        return None
    return full_path


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("signin", next=request.path))
        return view(*args, **kwargs)

    return wrapped


@app.context_processor
def inject_user():
    return {
        "current_user_name": session.get("user_name"),
        "is_authenticated": "user_id" in session,
    }


@app.route("/")
def index():
    error = request.args.get("error")
    selected_filename = request.args.get("filename")
    message = request.args.get("message")
    if message:
        flash(message, "success")
    return render_template("index.html", error=error, selected_filename=selected_filename)


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return redirect(url_for("index", error="No file part in the request."))

    file = request.files["image"]
    if file.filename == "":
        return redirect(url_for("index", error="Please choose an image file to upload."))

    if not allowed_file(file.filename):
        return redirect(
            url_for(
                "index",
                error="Invalid file type. Only JPG, JPEG, and PNG images are allowed.",
                filename=file.filename,
            )
        )

    original_ext = file.filename.rsplit(".", 1)[1].lower()
    if original_ext == "jpeg":
        original_ext = "jpg"

    unique_id = uuid.uuid4().hex
    original_name = "%s.%s" % (unique_id, original_ext)
    result_name = "%s_colorized.%s" % (unique_id, original_ext)

    original_path = safe_join(UPLOAD_FOLDER, original_name)
    result_path = safe_join(RESULT_FOLDER, result_name)
    sketch_name = "%s_sketch.%s" % (unique_id, original_ext)
    sketch_path = safe_join(RESULT_FOLDER, sketch_name)
    if original_path is None or result_path is None or sketch_path is None:
        return redirect(url_for("index", error="Could not prepare storage paths.", filename=file.filename))

    try:
        file.save(original_path)
    except Exception:
        return redirect(url_for("index", error="Failed to save the uploaded file.", filename=file.filename))

    try:
        import cv2

        test_img = cv2.imread(original_path, cv2.IMREAD_COLOR)
    except Exception:
        test_img = None

    if test_img is None:
        try:
            os.remove(original_path)
        except OSError:
            pass
        return redirect(url_for("index", error="The uploaded file is not a valid image.", filename=file.filename))

    try:
        colorizer.colorize_image(original_path, result_path)
    except Exception:
        for p in (original_path, result_path, sketch_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass
        return redirect(
            url_for("index", error="Image colorization failed. Please try another image.", filename=file.filename)
        )

    try:
        sketcher.generate_sketch(original_path, sketch_path)
    except Exception:
        for p in (original_path, result_path, sketch_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass
        return redirect(
            url_for(
                "index", error="Pencil sketch generation failed. Please try another image.", filename=file.filename
            )
        )

    return render_template(
        "result.html",
        original_image=url_for("static", filename="uploads/%s" % original_name),
        colorized_image=url_for("static", filename="results/%s" % result_name),
        sketch_image=url_for("static", filename="results/%s" % sketch_name),
        colorized_filename=result_name,
        sketch_filename=sketch_name,
    )


@app.route("/download/<path:filename>")
def download(filename):
    safe_path = safe_join(RESULT_FOLDER, filename)
    if safe_path is None or not os.path.isfile(safe_path):
        abort(404)
    base = os.path.basename(safe_path)
    ext = os.path.splitext(base)[1] or ".jpg"
    if "sketch" in base:
        download_name = "chromora_sketch" + ext
    elif "colorized" in base:
        download_name = "chromora_colorized" + ext
    else:
        download_name = base
    return send_from_directory(RESULT_FOLDER, base, as_attachment=True, download_name=download_name)


# ---------- Authentication ----------


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_template("signup.html")

    full_name = request.form.get("full_name", "").strip()
    email = request.form.get("email", "").strip()
    password = request.form.get("password", "")
    confirm_password = request.form.get("confirm_password", "")

    error = None
    if len(full_name) < 2:
        error = "Please enter your full name (at least 2 characters)."
    elif "@" not in email or "." not in email.split("@")[-1]:
        error = "Please enter a valid email address."
    elif len(password) < 6:
        error = "Password must be at least 6 characters long."
    elif password != confirm_password:
        error = "Passwords do not match."

    if error:
        return render_template(
            "signup.html", error=error, full_name=full_name, email=email
        )

    if database.get_user_by_email(email):
        return render_template(
            "signup.html",
            error="An account with this email already exists.",
            full_name=full_name,
            email=email,
        )

    password_hash = generate_password_hash(password)
    user_id = database.create_user(full_name, email, password_hash)
    if user_id is None:
        return render_template(
            "signup.html",
            error="An account with this email already exists.",
            full_name=full_name,
            email=email,
        )

    flash("Account created successfully. Please sign in.", "success")
    return redirect(url_for("signin"))


@app.route("/signin", methods=["GET", "POST"])
def signin():
    next_url = request.values.get("next", "")

    if request.method == "GET":
        return render_template("signin.html", next_url=next_url)

    email = request.form.get("email", "").strip()
    password = request.form.get("password", "")
    next_url = request.form.get("next_url") or next_url

    user = database.get_user_by_email(email)
    if user is None or not check_password_hash(user["password_hash"], password):
        return render_template(
            "signin.html",
            error="Invalid email or password.",
            email=email,
            next_url=next_url,
        )

    session["user_id"] = user["id"]
    session["user_name"] = user["full_name"]
    flash("Welcome back, %s!" % user["full_name"], "success")

    if next_url and isinstance(next_url, str) and next_url.startswith("/"):
        return redirect(next_url)
    return redirect(url_for("index"))


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("index"))


@app.route("/account")
@login_required
def account():
    user = database.get_user_by_id(session["user_id"])
    if user is None:
        session.clear()
        flash("Please sign in to continue.", "error")
        return redirect(url_for("signin"))
    return render_template("account.html", user=user)


@app.errorhandler(413)
def too_large(_e):
    return redirect(url_for("index", error="File is too large. Maximum size is 5 MB."))


@app.errorhandler(404)
def not_found(_e):
    return render_template("index.html", error="The requested page was not found."), 404


@app.errorhandler(500)
def server_error(_e):
    return render_template("index.html", error="An internal error occurred. Please try again."), 500


if __name__ == "__main__":
    if not colorizer.is_model_loaded():
        print("WARNING: Colorization model failed to load:")
        print(colorizer.get_model_error())
    import socket

    port = 5000
    for candidate in range(5000, 5010):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("0.0.0.0", candidate))
                port = candidate
                break
            except OSError:
                continue
    print("Starting Chromora AI on http://127.0.0.1:%d" % port)
    app.run(host="0.0.0.0", port=port, debug=False)
