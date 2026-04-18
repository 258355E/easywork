# CV Template Refactor — Complete Restructured Code

Below is a production-style refactor that fixes the mismatch between **template selection**, **editor preview**, and **final result** by introducing **one shared CV renderer**.

---

## Goal

Use the same CV HTML structure everywhere:

* template selection preview
* editor preview
* final result page
* future PDF export

The canonical source remains:

* `resume_data`
* `saved_template`
* `theme_color`

Stored in the database.

---

## Recommended folder structure

```text
project/
  app.py
  templates/
    base.html
    index.html
    improve.html
    documents.html
    select_template.html
    edit.html
    result.html
    cv/
      _base_cv.html
      _resume_sections.html
      classic.html
      modern.html
      sidebar.html
      twocol.html
      executive.html
      creative.html
  static/
    css/
      cv.css
      app.css
```

---

## 1) `app.py`

Replace the rendering-related parts with this structure.

````python
import os
import re
import time
import uuid
import json
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv
from flask import (
    Flask,
    redirect,
    render_template,
    session,
    url_for,
    request,
    jsonify,
)
from authlib.integrations.flask_client import OAuth
from werkzeug.utils import secure_filename
from google import genai
from flask_sqlalchemy import SQLAlchemy

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, ".env"))

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-this-secret-key")

app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = False

UPLOAD_FOLDER = os.path.join(basedir, "uploads")
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"pdf", "doc", "docx", "txt", "html"}
MAX_FILE_SIZE = 10 * 1024 * 1024
MODEL_ID = "gemini-2.5-flash"

ALLOWED_TEMPLATES = {
    "classic",
    "modern",
    "sidebar",
    "twocol",
    "executive",
    "creative",
}
HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")

DEFAULT_TEMPLATE = "classic"
DEFAULT_THEME = "#ba7517"


google_client_id = os.getenv("GOOGLE_CLIENT_ID")
google_client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
google_api_key = os.getenv("GOOGLE_API_KEY")

db_host = os.getenv("DB_HOST", "127.0.0.1")
db_port = os.getenv("DB_PORT", "3306")
db_name = os.getenv("DB_NAME", "test")
db_user = os.getenv("DB_USER", "root")
db_password = quote_plus(os.getenv("DB_PASSWORD", ""))

if not google_client_id:
    raise ValueError("GOOGLE_CLIENT_ID is missing in .env")
if not google_client_secret:
    raise ValueError("GOOGLE_CLIENT_SECRET is missing in .env")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is missing in .env")

app.config["SQLALCHEMY_DATABASE_URI"] = (
    f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
oauth = OAuth(app)

google = oauth.register(
    name="google",
    client_id=google_client_id,
    client_secret=google_client_secret,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

gemini_client = genai.Client(api_key=google_api_key)


class ResumeResult(db.Model):
    __tablename__ = "resume_results"

    id = db.Column(db.Integer, primary_key=True)
    result_id = db.Column(db.String(64), unique=True, nullable=False, index=True)

    user_google_id = db.Column(db.String(255), nullable=True)
    user_name = db.Column(db.String(255), nullable=True)
    user_email = db.Column(db.String(255), nullable=True, index=True)
    user_picture = db.Column(db.Text, nullable=True)

    original_filename = db.Column(db.String(255), nullable=False)
    saved_path = db.Column(db.Text, nullable=False)

    jd_text = db.Column(db.Text, nullable=True)
    resume_data = db.Column(db.JSON, nullable=False)

    theme_color = db.Column(db.String(20), nullable=False, default=DEFAULT_THEME)
    saved_template = db.Column(db.String(32), nullable=True, default=DEFAULT_TEMPLATE)

    created_at = db.Column(db.DateTime, server_default=db.func.now(), nullable=False)
    updated_at = db.Column(
        db.DateTime,
        server_default=db.func.now(),
        onupdate=db.func.now(),
        nullable=False,
    )


@app.before_request
def create_tables():
    if not getattr(app, "_db_ready", False):
        db.create_all()
        app._db_ready = True


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def wait_for_gemini_file(file_obj, timeout=60):
    start = time.time()
    while getattr(file_obj, "state", None) and file_obj.state.name == "PROCESSING":
        if time.time() - start > timeout:
            raise TimeoutError("Gemini file processing timed out.")
        time.sleep(2)
        file_obj = gemini_client.files.get(name=file_obj.name)
    return file_obj


def clean_json_response(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def ensure_list(value):
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [str(value)]


def normalize_resume_data(data):
    if not isinstance(data, dict):
        data = {}

    normalized = {
        "full_name": str(data.get("full_name", "") or ""),
        "headline": str(data.get("headline", "") or ""),
        "career_summary": str(data.get("career_summary", "") or ""),
        "contact": {"phone": "", "email": "", "linkedin": "", "address": ""},
        "work_experience": [],
        "education": [],
        "certifications": [],
        "skills": {"software": [], "languages": [], "other": []},
        "suggestions": [],
    }

    contact = data.get("contact")
    if isinstance(contact, dict):
        normalized["contact"]["phone"] = str(contact.get("phone", "") or "")
        normalized["contact"]["email"] = str(contact.get("email", "") or "")
        normalized["contact"]["linkedin"] = str(contact.get("linkedin", "") or "")
        normalized["contact"]["address"] = str(contact.get("address", "") or "")

    for job in ensure_list(data.get("work_experience")):
        if not isinstance(job, dict):
            continue
        normalized["work_experience"].append(
            {
                "job_title": str(job.get("job_title", "") or ""),
                "company": str(job.get("company", "") or ""),
                "location": str(job.get("location", "") or ""),
                "start_date": str(job.get("start_date", "") or ""),
                "end_date": str(job.get("end_date", "") or ""),
                "bullets": [
                    str(x) for x in ensure_list(job.get("bullets")) if str(x).strip()
                ],
            }
        )

    for edu in ensure_list(data.get("education")):
        if not isinstance(edu, dict):
            continue
        normalized["education"].append(
            {
                "institution": str(edu.get("institution", "") or ""),
                "degree": str(edu.get("degree", "") or ""),
                "location": str(edu.get("location", "") or ""),
                "graduation_date": str(edu.get("graduation_date", "") or ""),
                "details": [
                    str(x) for x in ensure_list(edu.get("details")) if str(x).strip()
                ],
            }
        )

    for cert in ensure_list(data.get("certifications")):
        if not isinstance(cert, dict):
            continue
        normalized["certifications"].append(
            {
                "name": str(cert.get("name", "") or ""),
                "issuer": str(cert.get("issuer", "") or ""),
                "date": str(cert.get("date", "") or ""),
            }
        )

    skills = data.get("skills")
    if isinstance(skills, dict):
        normalized["skills"]["software"] = [
            str(x) for x in ensure_list(skills.get("software")) if str(x).strip()
        ]
        normalized["skills"]["languages"] = [
            str(x) for x in ensure_list(skills.get("languages")) if str(x).strip()
        ]
        normalized["skills"]["other"] = [
            str(x) for x in ensure_list(skills.get("other")) if str(x).strip()
        ]

    normalized["suggestions"] = [
        str(x) for x in ensure_list(data.get("suggestions")) if str(x).strip()
    ]

    return normalized


def validate_theme_color(color: str) -> str:
    color = str(color).strip().lower()
    if color and not color.startswith("#"):
        color = "#" + color
    if HEX_COLOR_RE.match(color):
        return color
    return DEFAULT_THEME


def validate_template(template: str) -> str:
    if template in ALLOWED_TEMPLATES:
        return template
    return DEFAULT_TEMPLATE


def get_resume_or_404(result_id: str):
    return ResumeResult.query.filter_by(result_id=result_id).first()


def enforce_owner_or_403(result: ResumeResult):
    user = session.get("user")
    if not user:
        return jsonify({"ok": False, "error": "Please log in first."}), 401
    if result.user_google_id and result.user_google_id != user.get("id"):
        return jsonify({"ok": False, "error": "Forbidden."}), 403
    return None


def build_cv_context(result: ResumeResult):
    template = validate_template(result.saved_template or DEFAULT_TEMPLATE)
    theme = validate_theme_color(result.theme_color or DEFAULT_THEME)
    cv = normalize_resume_data(result.resume_data)
    return {
        "cv": cv,
        "template": template,
        "theme": theme,
        "result": result,
        "result_id": result.result_id,
        "user": session.get("user"),
    }


@app.route("/")
def index():
    return render_template("index.html", user=session.get("user"))


@app.route("/login")
def login():
    redirect_uri = url_for("auth_callback", _external=True)
    return google.authorize_redirect(redirect_uri)


@app.route("/callback")
def auth_callback():
    token = google.authorize_access_token()
    user_info = token.get("userinfo")
    if not user_info:
        user_info = google.get("userinfo").json()

    session["user"] = {
        "id": user_info.get("sub"),
        "name": user_info.get("name"),
        "email": user_info.get("email"),
        "picture": user_info.get("picture"),
    }

    existing_count = ResumeResult.query.filter_by(
        user_google_id=session["user"].get("id")
    ).count()
    if existing_count > 0:
        return redirect(url_for("documents"))
    return redirect(url_for("improve"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/improve")
def improve():
    user = session.get("user")
    if not user:
        return redirect(url_for("login"))
    return render_template("improve.html", user=user)


@app.route("/select-template/<result_id>")
def select_template(result_id):
    result = get_resume_or_404(result_id)
    if not result:
        return "Result not found", 404

    context = build_cv_context(result)
    return render_template("select_template.html", **context)


@app.route("/edit/<result_id>")
def edit_page(result_id):
    user = session.get("user")
    if not user:
        return redirect(url_for("login"))

    result = get_resume_or_404(result_id)
    if not result:
        return "Result not found", 404

    template = validate_template(
        request.args.get("template", "").strip() or result.saved_template or DEFAULT_TEMPLATE
    )
    theme = validate_theme_color(
        request.args.get("color", "").strip() or result.theme_color or DEFAULT_THEME
    )

    changed = False
    if result.saved_template != template:
        result.saved_template = template
        changed = True
    if result.theme_color != theme:
        result.theme_color = theme
        changed = True
    if changed:
        db.session.commit()

    context = build_cv_context(result)
    return render_template("edit.html", **context)


@app.route("/result/<result_id>")
def result_page(result_id):
    result = get_resume_or_404(result_id)
    if not result:
        return "Result not found", 404

    template = validate_template(
        request.args.get("template", "").strip() or result.saved_template or DEFAULT_TEMPLATE
    )
    theme = validate_theme_color(
        request.args.get("color", "").strip() or result.theme_color or DEFAULT_THEME
    )

    changed = False
    if result.saved_template != template:
        result.saved_template = template
        changed = True
    if result.theme_color != theme:
        result.theme_color = theme
        changed = True
    if changed:
        db.session.commit()

    context = build_cv_context(result)
    return render_template("result.html", **context)


@app.route("/documents")
def documents():
    user = session.get("user")
    if not user:
        return redirect(url_for("login"))

    items = (
        ResumeResult.query.filter_by(user_google_id=user.get("id"))
        .order_by(ResumeResult.updated_at.desc())
        .all()
    )
    return render_template("documents.html", user=user, items=items)


@app.route("/api/result/<result_id>")
def get_result_api(result_id):
    result = get_resume_or_404(result_id)
    if not result:
        return jsonify({"ok": False, "error": "Result not found."}), 404

    owner_check = enforce_owner_or_403(result)
    if owner_check:
        return owner_check

    return jsonify(
        {
            "ok": True,
            "result_id": result.result_id,
            "resume_data": normalize_resume_data(result.resume_data),
            "theme_color": validate_theme_color(result.theme_color),
            "saved_template": validate_template(result.saved_template),
        }
    )


@app.route("/api/result/<result_id>/update", methods=["POST"])
def update_result_api(result_id):
    result = get_resume_or_404(result_id)
    if not result:
        return jsonify({"ok": False, "error": "Result not found."}), 404

    owner_check = enforce_owner_or_403(result)
    if owner_check:
        return owner_check

    payload = request.get_json(silent=True) or {}
    resume_data = normalize_resume_data(payload.get("resume_data") or {})
    theme_color = validate_theme_color(payload.get("theme_color") or result.theme_color)
    saved_template = validate_template(payload.get("saved_template") or result.saved_template)

    result.resume_data = resume_data
    result.theme_color = theme_color
    result.saved_template = saved_template
    db.session.commit()

    return jsonify({"ok": True})


@app.route("/api/result/<result_id>/delete", methods=["DELETE"])
def delete_result_api(result_id):
    result = get_resume_or_404(result_id)
    if not result:
        return jsonify({"ok": False, "error": "Result not found."}), 404

    owner_check = enforce_owner_or_403(result)
    if owner_check:
        return owner_check

    db.session.delete(result)
    db.session.commit()
    return jsonify({"ok": True})


@app.route("/api/user-cvs")
def user_cvs_api():
    user = session.get("user")
    if not user:
        return jsonify({"ok": False, "error": "Please log in first."}), 401

    items = (
        ResumeResult.query.filter_by(user_google_id=user.get("id"))
        .order_by(ResumeResult.updated_at.desc())
        .limit(20)
        .all()
    )

    return jsonify(
        {
            "ok": True,
            "items": [
                {
                    "result_id": x.result_id,
                    "full_name": (x.resume_data or {}).get("full_name", ""),
                    "headline": (x.resume_data or {}).get("headline", ""),
                    "template": validate_template(x.saved_template),
                    "theme_color": validate_theme_color(x.theme_color),
                    "updated_at": x.updated_at.isoformat() if x.updated_at else None,
                }
                for x in items
            ],
        }
    )


@app.route("/api/reuse-cv", methods=["POST"])
def reuse_cv_api():
    user = session.get("user")
    if not user:
        return jsonify({"ok": False, "error": "Please log in first."}), 401

    payload = request.get_json(silent=True) or {}
    reuse_id = str(payload.get("reuse_result_id", "")).strip()
    jd_text = str(payload.get("jd_text", "")).strip()

    if not reuse_id:
        return jsonify({"ok": False, "error": "reuse_result_id is required."}), 400

    existing = get_resume_or_404(reuse_id)
    if not existing:
        return jsonify({"ok": False, "error": "Previous CV not found."}), 404

    if existing.user_google_id and existing.user_google_id != user.get("id"):
        return jsonify({"ok": False, "error": "Forbidden."}), 403

    existing_json = json.dumps(normalize_resume_data(existing.resume_data), ensure_ascii=False)
    prompt = f"""
You are an expert ATS resume writer.

Below is an existing CV in JSON format. Rewrite and improve it — strengthen the language,
fix structure, and tailor it to the provided job description if available.

IMPORTANT RULES:
- Return ONLY valid JSON.
- Do not wrap the JSON in markdown or triple backticks.
- Keep wording concise and achievement-focused.
- Preserve truthfulness. Do not invent experience.

Existing CV JSON:
{existing_json}

Job Description:
{jd_text if jd_text else 'No job description provided.'}

Return JSON in exactly this structure:
{{
  "full_name": "",
  "headline": "",
  "contact": {{"phone": "", "email": "", "linkedin": "", "address": ""}},
  "career_summary": "",
  "work_experience": [{{"job_title": "", "company": "", "location": "", "start_date": "", "end_date": "", "bullets": [""]}}],
  "education": [{{"institution": "", "degree": "", "location": "", "graduation_date": "", "details": [""]}}],
  "certifications": [{{"name": "", "issuer": "", "date": ""}}],
  "skills": {{"software": [""], "languages": [""], "other": [""]}},
  "suggestions": [""]
}}
"""

    try:
        response = gemini_client.models.generate_content(model=MODEL_ID, contents=[prompt])
        cleaned_text = clean_json_response(response.text or "")
        resume_data = normalize_resume_data(json.loads(cleaned_text))

        result_id = uuid.uuid4().hex
        new_result = ResumeResult(
            result_id=result_id,
            user_google_id=user.get("id"),
            user_name=user.get("name"),
            user_email=user.get("email"),
            user_picture=user.get("picture"),
            original_filename=existing.original_filename,
            saved_path="reused",
            jd_text=jd_text,
            resume_data=resume_data,
            theme_color=existing.theme_color or DEFAULT_THEME,
            saved_template=existing.saved_template or DEFAULT_TEMPLATE,
        )
        db.session.add(new_result)
        db.session.commit()

        return jsonify(
            {
                "ok": True,
                "result_id": result_id,
                "redirect_url": url_for("select_template", result_id=result_id),
            }
        )
    except json.JSONDecodeError:
        return jsonify({"ok": False, "error": "Gemini returned invalid JSON. Please try again."}), 500
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/improve-cv", methods=["POST"])
def improve_cv_api():
    user = session.get("user")
    if not user:
        return jsonify({"ok": False, "error": "Please log in first."}), 401

    jd_text = request.form.get("jd_text", "").strip()

    if "cv_file" not in request.files:
        return jsonify({"ok": False, "error": "CV file is required."}), 400

    cv_file = request.files["cv_file"]
    if cv_file.filename == "":
        return jsonify({"ok": False, "error": "No file selected."}), 400
    if not allowed_file(cv_file.filename):
        return jsonify({"ok": False, "error": "Unsupported CV file type."}), 400

    cv_file.seek(0, os.SEEK_END)
    size = cv_file.tell()
    cv_file.seek(0)
    if size > MAX_FILE_SIZE:
        return jsonify({"ok": False, "error": "CV file must be under 10 MB."}), 400

    safe_name = secure_filename(cv_file.filename)
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    saved_path = os.path.join(UPLOAD_FOLDER, unique_name)
    cv_file.save(saved_path)

    try:
        uploaded_cv = gemini_client.files.upload(file=saved_path)
        uploaded_cv = wait_for_gemini_file(uploaded_cv)

        if not getattr(uploaded_cv, "state", None) or uploaded_cv.state.name != "ACTIVE":
            state_name = getattr(getattr(uploaded_cv, "state", None), "name", "UNKNOWN")
            return jsonify({"ok": False, "error": f"Gemini file processing failed: {state_name}"}), 500

        prompt = f"""
You are an expert ATS resume writer.

Read the uploaded CV and the user details. Rewrite the resume into a clean, ATS-friendly format.

IMPORTANT RULES:
- Return ONLY valid JSON.
- Do not wrap the JSON in markdown or triple backticks.
- Do not include explanations outside JSON.
- Keep wording concise and achievement-focused.
- Use strong action verbs.
- Tailor the content to the provided job description if available.
- Preserve truthfulness. Do not invent experience.

User details:
- Name: {user.get('name', '')}
- Email: {user.get('email', '')}

Job Description:
{jd_text if jd_text else 'No job description provided.'}

Return JSON in exactly this structure:
{{
  "full_name": "",
  "headline": "",
  "contact": {{"phone": "", "email": "", "linkedin": "", "address": ""}},
  "career_summary": "",
  "work_experience": [{{"job_title": "", "company": "", "location": "", "start_date": "", "end_date": "", "bullets": [""]}}],
  "education": [{{"institution": "", "degree": "", "location": "", "graduation_date": "", "details": [""]}}],
  "certifications": [{{"name": "", "issuer": "", "date": ""}}],
  "skills": {{"software": [""], "languages": [""], "other": [""]}},
  "suggestions": [""]
}}
"""

        response = gemini_client.models.generate_content(
            model=MODEL_ID,
            contents=[uploaded_cv, prompt],
        )

        cleaned_text = clean_json_response(response.text or "")
        resume_data = normalize_resume_data(json.loads(cleaned_text))
        result_id = uuid.uuid4().hex

        new_result = ResumeResult(
            result_id=result_id,
            user_google_id=user.get("id"),
            user_name=user.get("name"),
            user_email=user.get("email"),
            user_picture=user.get("picture"),
            original_filename=cv_file.filename,
            saved_path=saved_path,
            jd_text=jd_text,
            resume_data=resume_data,
            theme_color=DEFAULT_THEME,
            saved_template=DEFAULT_TEMPLATE,
        )
        db.session.add(new_result)
        db.session.commit()

        try:
            os.remove(saved_path)
        except OSError:
            pass

        return jsonify(
            {
                "ok": True,
                "result_id": result_id,
                "redirect_url": url_for("select_template", result_id=result_id),
            }
        )
    except json.JSONDecodeError:
        return jsonify({"ok": False, "error": "Gemini returned invalid JSON. Please try again."}), 500
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500




