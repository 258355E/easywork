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
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.exceptions import RequestEntityTooLarge
from google import genai
from flask_sqlalchemy import SQLAlchemy


basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, ".env"))

app = Flask(__name__)


def env_truthy(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


IS_DEBUG = env_truthy("FLASK_DEBUG", default=False) or os.getenv("FLASK_ENV") == "development"

secret = os.getenv("FLASK_SECRET_KEY")
if not secret:
    if IS_DEBUG:
        # Stable secrets are required for multi-instance SaaS; for local dev, keep it simple.
        secret = "dev-" + uuid.uuid4().hex
    else:
        raise ValueError("FLASK_SECRET_KEY is required in production.")
app.secret_key = secret

app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = env_truthy("SESSION_COOKIE_SECURE", default=not IS_DEBUG)

UPLOAD_FOLDER = os.path.join(basedir, "uploads")
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"pdf", "doc", "docx", "txt", "html"}
MAX_FILE_SIZE = 10 * 1024 * 1024
MODEL_ID = "gemini-2.5-flash"

# Backstop upload limits at the server layer (in addition to manual size checks).
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE + (1024 * 1024)

if env_truthy("TRUST_PROXY", default=False):
    # For deployments behind a reverse proxy (Render/NGINX/etc.).
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

ALLOWED_TEMPLATES = {
    "classic",
    "modern",
    "sidebar",
    "twocol",
    "executive",
    "creative",
    "atsclean",
    "atscompact",
    "minimal",
    "hybrid",
    "impact",
}
DEFAULT_TEMPLATE = "classic"
DEFAULT_THEME = "#ba7517"

HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")

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


class Purchase(db.Model):
    __tablename__ = "purchases"

    id = db.Column(db.Integer, primary_key=True)
    user_google_id = db.Column(db.String(255), nullable=False, index=True)
    result_id = db.Column(db.String(64), nullable=False, index=True)
    status = db.Column(db.String(32), nullable=False, default="paid")

    created_at = db.Column(db.DateTime, server_default=db.func.now(), nullable=False)

    __table_args__ = (
        db.UniqueConstraint("user_google_id", "result_id", name="uq_purchase_user_result"),
    )


@app.before_request
def create_tables():
    if not getattr(app, "_db_ready", False):
        db.create_all()
        app._db_ready = True


@app.after_request
def add_security_headers(response):
    # Minimal hardening that shouldn't break existing templates.
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Permissions-Policy", "geolocation=()")

    # CV pages contain sensitive personal data; discourage caching.
    if request.path.startswith(("/result/", "/edit/", "/select-template/")):
        response.headers.setdefault("Cache-Control", "no-store, max-age=0")
        response.headers.setdefault("Pragma", "no-cache")
    return response


@app.errorhandler(RequestEntityTooLarge)
def handle_request_too_large(_e):
    if request.path.startswith("/api/"):
        return jsonify({"ok": False, "error": "File too large."}), 413
    return "File too large.", 413


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
    text = (text or "").strip()
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
    return [value]


def normalize_resume_data(data):
    if not isinstance(data, dict):
        data = {}

    def clamp_float(value, default, minimum, maximum):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default
        return max(minimum, min(maximum, value))

    normalized = {
        "full_name": str(data.get("full_name", "") or ""),
        "headline": str(data.get("headline", "") or ""),
        "career_summary": str(data.get("career_summary", "") or ""),
        "contact": {
            "phone": "",
            "email": "",
            "linkedin": "",
            "address": "",
        },
        "work_experience": [],
        "education": [],
        "certifications": [],
        "skills": {
            "software": [],
            "languages": [],
            "other": [],
        },
        "suggestions": [],
        "layout": {
            "font_scale": 1.0,
            "line_spacing": 1.0,
            "space_scale": 1.0,
        },
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
                    str(x).strip()
                    for x in ensure_list(job.get("bullets"))
                    if str(x).strip()
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
                    str(x).strip()
                    for x in ensure_list(edu.get("details"))
                    if str(x).strip()
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
            str(x).strip()
            for x in ensure_list(skills.get("software"))
            if str(x).strip()
        ]
        normalized["skills"]["languages"] = [
            str(x).strip()
            for x in ensure_list(skills.get("languages"))
            if str(x).strip()
        ]
        normalized["skills"]["other"] = [
            str(x).strip()
            for x in ensure_list(skills.get("other"))
            if str(x).strip()
        ]

    normalized["suggestions"] = [
        str(x).strip()
        for x in ensure_list(data.get("suggestions"))
        if str(x).strip()
    ]

    layout = data.get("layout")
    if isinstance(layout, dict):
        normalized["layout"]["font_scale"] = clamp_float(
            layout.get("font_scale"), 1.0, 0.9, 1.12
        )
        normalized["layout"]["line_spacing"] = clamp_float(
            layout.get("line_spacing"), 1.0, 0.88, 1.2
        )
        normalized["layout"]["space_scale"] = clamp_float(
            layout.get("space_scale"), 1.0, 0.8, 1.2
        )

    return normalized


def validate_theme_color(color: str) -> str:
    color = str(color or "").strip().lower()
    if color and not color.startswith("#"):
        color = "#" + color
    if HEX_COLOR_RE.match(color):
        return color
    return DEFAULT_THEME


def validate_template(template: str) -> str:
    template = str(template or "").strip()
    if template in ALLOWED_TEMPLATES:
        return template
    return DEFAULT_TEMPLATE


def get_result_or_404(result_id: str):
    return ResumeResult.query.filter_by(result_id=result_id).first()


def require_login_json():
    user = session.get("user")
    if not user:
        return None, jsonify({"ok": False, "error": "Please log in first."}), 401
    return user, None, None


def require_login_page():
    user = session.get("user")
    if not user:
        return None, redirect(url_for("login"))
    return user, None


def is_paywall_enabled() -> bool:
    # Temporary testing override: keep PDF downloads unlocked without redirecting to payment.
    return False


def has_paid_access(user_google_id: str, result_id: str) -> bool:
    if not user_google_id or not result_id:
        return False
    paid = (
        Purchase.query.filter_by(user_google_id=user_google_id, result_id=result_id, status="paid")
        .limit(1)
        .first()
    )
    return paid is not None


def check_result_ownership(result: ResumeResult):
    user = session.get("user")
    if not user:
        return jsonify({"ok": False, "error": "Please log in first."}), 401
    # Lock down results strictly to the owner. "Ownerless" results are treated as forbidden.
    if not result.user_google_id or result.user_google_id != user.get("id"):
        return jsonify({"ok": False, "error": "Forbidden."}), 403
    return None


def build_resume_view_context(result: ResumeResult, template_override=None, color_override=None):
    template = validate_template(template_override or result.saved_template or DEFAULT_TEMPLATE)
    theme = validate_theme_color(color_override or result.theme_color or DEFAULT_THEME)
    cv = normalize_resume_data(result.resume_data)

    return {
        "user": session.get("user"),
        "result": result,
        "result_id": result.result_id,
        "cv": cv,
        "template": template,
        "theme": theme,
    }


def persist_template_and_theme(result: ResumeResult, template_value: str, theme_value: str):
    changed = False
    if result.saved_template != template_value:
        result.saved_template = template_value
        changed = True
    if result.theme_color != theme_value:
        result.theme_color = theme_value
        changed = True
    if changed:
        db.session.commit()


def build_resume_prompt(
    existing_json: str,
    jd_text: str,
    change_request: str = "",
    user_details: str = "",
) -> str:
    return f"""
You are an expert ATS resume writer.

Below is an existing CV in JSON format. Rewrite and improve it — strengthen the language,
fix structure, and tailor it to the provided job description if available.

IMPORTANT RULES:
- Return ONLY valid JSON.
- Do not wrap the JSON in markdown or triple backticks.
- Keep wording concise and achievement-focused.
- Preserve truthfulness. Do not invent experience.
- Follow the user's requested updates when they say what should change.
- Use additional personal details only when they are clear and resume-appropriate.

Existing CV JSON:
{existing_json}

Job Description:
{jd_text if jd_text else "No job description provided."}

Requested changes:
{change_request if change_request else "No specific change request provided."}

Additional user details:
{user_details if user_details else "No additional user details provided."}

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


def build_uploaded_cv_prompt(
    user: dict,
    jd_text: str,
    change_request: str = "",
    user_details: str = "",
) -> str:
    return f"""
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
- Follow the user's requested updates when they say what should change.
- Use additional personal details only when they are clear and resume-appropriate.

User details:
- Name: {user.get("name", "")}
- Email: {user.get("email", "")}

Job Description:
{jd_text if jd_text else "No job description provided."}

Requested changes:
{change_request if change_request else "No specific change request provided."}

Additional user details:
{user_details if user_details else "No additional user details provided."}

Return JSON in exactly this structure:
{{
  "full_name": "",
  "headline": "",
  "contact": {{
    "phone": "",
    "email": "",
    "linkedin": "",
    "address": ""
  }},
  "career_summary": "",
  "work_experience": [
    {{
      "job_title": "",
      "company": "",
      "location": "",
      "start_date": "",
      "end_date": "",
      "bullets": [""]
    }}
  ],
  "education": [
    {{
      "institution": "",
      "degree": "",
      "location": "",
      "graduation_date": "",
      "details": [""]
    }}
  ],
  "certifications": [
    {{
      "name": "",
      "issuer": "",
      "date": ""
    }}
  ],
  "skills": {{
    "software": [""],
    "languages": [""],
    "other": [""]
  }},
  "suggestions": [""]
}}
"""


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

    existing = ResumeResult.query.filter_by(
        user_google_id=session["user"].get("id")
    ).count()

    if existing > 0:
        return redirect(url_for("documents"))
    return redirect(url_for("improve"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/improve")
def improve():
    user, redirect_response = require_login_page()
    if redirect_response:
        return redirect_response
    return render_template("improve.html", user=user)


@app.route("/select-template/<result_id>")
def select_template(result_id):
    user, redirect_response = require_login_page()
    if redirect_response:
        return redirect_response

    result = get_result_or_404(result_id)
    if not result:
        return "Result not found", 404
    if not result.user_google_id or result.user_google_id != user.get("id"):
        return "Forbidden", 403

    template = validate_template(request.args.get("template", "").strip() or result.saved_template)
    color_raw = request.args.get("color", "").strip()
    theme = validate_theme_color(color_raw) if color_raw else validate_theme_color(result.theme_color)

    persist_template_and_theme(result, template, theme)

    context = build_resume_view_context(result)
    return render_template("select_template.html", **context)


@app.route("/edit/<result_id>")
def edit_page(result_id):
    user, redirect_response = require_login_page()
    if redirect_response:
        return redirect_response

    result = get_result_or_404(result_id)
    if not result:
        return "Result not found", 404
    if not result.user_google_id or result.user_google_id != user.get("id"):
        return "Forbidden", 403

    template = validate_template(request.args.get("template", "").strip() or result.saved_template)
    color_raw = request.args.get("color", "").strip()
    theme = validate_theme_color(color_raw) if color_raw else validate_theme_color(result.theme_color)

    persist_template_and_theme(result, template, theme)

    context = build_resume_view_context(result)
    return render_template("edit.html", **context)


@app.route("/result/<result_id>")
def result_page(result_id):
    user, redirect_response = require_login_page()
    if redirect_response:
        return redirect_response

    result = get_result_or_404(result_id)
    if not result:
        return "Result not found", 404
    if not result.user_google_id or result.user_google_id != user.get("id"):
        return "Forbidden", 403

    template = validate_template(request.args.get("template", "").strip() or result.saved_template)
    color_raw = request.args.get("color", "").strip()
    theme = validate_theme_color(color_raw) if color_raw else validate_theme_color(result.theme_color)

    persist_template_and_theme(result, template, theme)

    context = build_resume_view_context(result)
    paywall_enabled = is_paywall_enabled()
    is_paid = (not paywall_enabled) or has_paid_access(user.get("id"), result.result_id)
    context.update(
        {
            "paywall_enabled": paywall_enabled,
            "is_paid": is_paid,
            "pay_url": url_for("pay_page", result_id=result.result_id),
        }
    )
    return render_template("result.html", **context)


@app.route("/pay/<result_id>", methods=["GET"])
def pay_page(result_id):
    user, redirect_response = require_login_page()
    if redirect_response:
        return redirect_response

    result = get_result_or_404(result_id)
    if not result:
        return "Result not found", 404
    if not result.user_google_id or result.user_google_id != user.get("id"):
        return "Forbidden", 403

    # If already paid, send them back.
    if has_paid_access(user.get("id"), result.result_id):
        return redirect(url_for("result_page", result_id=result.result_id))

    payment_redirect_url = (os.getenv("PAYMENT_REDIRECT_URL") or "").strip()
    allow_test_payments = env_truthy("ALLOW_TEST_PAYMENTS", default=IS_DEBUG)
    price_label = (os.getenv("PAYMENT_PRICE_LABEL") or "LKR 1,000").strip()

    return render_template(
        "payment.html",
        user=user,
        result_id=result.result_id,
        price_label=price_label,
        payment_redirect_url=payment_redirect_url,
        allow_test_payments=allow_test_payments,
        return_url=url_for("result_page", result_id=result.result_id),
    )


@app.route("/pay/<result_id>/complete", methods=["POST"])
def pay_complete(result_id):
    user, redirect_response = require_login_page()
    if redirect_response:
        return redirect_response

    if not env_truthy("ALLOW_TEST_PAYMENTS", default=IS_DEBUG):
        return "Not available.", 404

    result = get_result_or_404(result_id)
    if not result:
        return "Result not found", 404
    if not result.user_google_id or result.user_google_id != user.get("id"):
        return "Forbidden", 403

    existing = (
        Purchase.query.filter_by(user_google_id=user.get("id"), result_id=result.result_id)
        .limit(1)
        .first()
    )
    if not existing:
        db.session.add(Purchase(user_google_id=user.get("id"), result_id=result.result_id, status="paid"))
        db.session.commit()

    return redirect(url_for("result_page", result_id=result.result_id))


@app.route("/documents")
def documents():
    user, redirect_response = require_login_page()
    if redirect_response:
        return redirect_response

    items = (
        ResumeResult.query.filter_by(user_google_id=user.get("id"))
        .order_by(ResumeResult.updated_at.desc())
        .all()
    )

    return render_template("documents.html", user=user, items=items, resumes=items)


@app.route("/healthz")
def healthz():
    return jsonify({"ok": True, "service": "cv-saas"})


@app.route("/api/result/<result_id>")
def get_result_api(result_id):
    result = get_result_or_404(result_id)
    if not result:
        return jsonify({"ok": False, "error": "Result not found."}), 404

    owner_error = check_result_ownership(result)
    if owner_error:
        return owner_error

    return jsonify(
        {
            "ok": True,
            "result_id": result.result_id,
            "resume_data": normalize_resume_data(result.resume_data),
            "theme_color": validate_theme_color(result.theme_color),
            "saved_template": validate_template(result.saved_template),
        }
    )


@app.route("/api/result/<result_id>/preview", methods=["POST"])
def preview_result_api(result_id):
    result = get_result_or_404(result_id)
    if not result:
        return jsonify({"ok": False, "error": "Result not found."}), 404

    owner_error = check_result_ownership(result)
    if owner_error:
        return owner_error

    payload = request.get_json(silent=True) or {}

    resume_data = normalize_resume_data(payload.get("resume_data") or result.resume_data)
    theme_color = validate_theme_color(payload.get("theme_color") or result.theme_color)
    saved_template = validate_template(payload.get("saved_template") or result.saved_template)

    try:
        html = render_template(
            f"cv/{saved_template}.html",
            cv=resume_data,
            template=saved_template,
            theme=theme_color,
        )
        return jsonify({"ok": True, "html": html})
    except Exception as e:
        return jsonify({"ok": False, "error": f"Preview render failed: {str(e)}"}), 500


@app.route("/api/result/<result_id>/update", methods=["POST"])
def update_result_api(result_id):
    result = get_result_or_404(result_id)
    if not result:
        return jsonify({"ok": False, "error": "Result not found."}), 404

    owner_error = check_result_ownership(result)
    if owner_error:
        return owner_error

    payload = request.get_json(silent=True) or {}

    resume_data = normalize_resume_data(payload.get("resume_data") or result.resume_data)
    theme_color = validate_theme_color(payload.get("theme_color") or result.theme_color)
    saved_template = validate_template(payload.get("saved_template") or result.saved_template)

    result.resume_data = resume_data
    result.theme_color = theme_color
    result.saved_template = saved_template
    db.session.commit()

    return jsonify({"ok": True})


@app.route("/api/result/<result_id>/delete", methods=["DELETE"])
def delete_result_api(result_id):
    result = get_result_or_404(result_id)
    if not result:
        return jsonify({"ok": False, "error": "Result not found."}), 404

    owner_error = check_result_ownership(result)
    if owner_error:
        return owner_error

    db.session.delete(result)
    db.session.commit()
    return jsonify({"ok": True})


@app.route("/api/user-cvs")
def user_cvs_api():
    user, error_response, status = require_login_json()
    if error_response:
        return error_response, status

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
                    "result_id": item.result_id,
                    "full_name": normalize_resume_data(item.resume_data).get("full_name", ""),
                    "headline": normalize_resume_data(item.resume_data).get("headline", ""),
                    "template": validate_template(item.saved_template),
                    "theme_color": validate_theme_color(item.theme_color),
                    "updated_at": item.updated_at.isoformat() if item.updated_at else None,
                }
                for item in items
            ],
        }
    )


@app.route("/api/reuse-cv", methods=["POST"])
def reuse_cv_api():
    user, error_response, status = require_login_json()
    if error_response:
        return error_response, status

    payload = request.get_json(silent=True) or {}
    reuse_id = str(payload.get("reuse_result_id", "")).strip()
    jd_text = str(payload.get("jd_text", "")).strip()
    change_request = str(payload.get("change_request", "")).strip()
    user_details = str(payload.get("user_details", "")).strip()

    if not reuse_id:
        return jsonify({"ok": False, "error": "reuse_result_id is required."}), 400

    existing = get_result_or_404(reuse_id)
    if not existing:
        return jsonify({"ok": False, "error": "Previous CV not found."}), 404

    if not existing.user_google_id or existing.user_google_id != user.get("id"):
        return jsonify({"ok": False, "error": "Forbidden."}), 403

    existing_json = json.dumps(normalize_resume_data(existing.resume_data), ensure_ascii=False)
    prompt = build_resume_prompt(existing_json, jd_text, change_request, user_details)

    try:
        response = gemini_client.models.generate_content(model=MODEL_ID, contents=[prompt])
        raw_text = response.text or ""
        cleaned_text = clean_json_response(raw_text)
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
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/improve-cv", methods=["POST"])
def improve_cv_api():
    user, error_response, status = require_login_json()
    if error_response:
        return error_response, status

    jd_text = request.form.get("jd_text", "").strip()
    change_request = request.form.get("change_request", "").strip()
    user_details = request.form.get("user_details", "").strip()

    reuse_id = request.form.get("reuse_result_id", "").strip()
    if reuse_id:
        existing = get_result_or_404(reuse_id)
        if not existing:
            return jsonify({"ok": False, "error": "Previous CV not found."}), 404
        if not existing.user_google_id or existing.user_google_id != user.get("id"):
            return jsonify({"ok": False, "error": "Forbidden."}), 403

        existing_json = json.dumps(normalize_resume_data(existing.resume_data), ensure_ascii=False)
        prompt = build_resume_prompt(existing_json, jd_text, change_request, user_details)

        try:
            response = gemini_client.models.generate_content(model=MODEL_ID, contents=[prompt])
            raw_text = response.text or ""
            cleaned_text = clean_json_response(raw_text)
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
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

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

        prompt = build_uploaded_cv_prompt(user, jd_text, change_request, user_details)

        response = gemini_client.models.generate_content(
            model=MODEL_ID,
            contents=[uploaded_cv, prompt],
        )

        raw_text = response.text or ""
        cleaned_text = clean_json_response(raw_text)
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
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# @app.route("/api/result/<result_id>/preview", methods=["POST"])
# def preview_result_api(result_id):
#     result = get_result_or_404(result_id)
#     if not result:
#         return jsonify({"ok": False, "error": "Result not found."}), 404

#     owner_error = check_result_ownership(result)
#     if owner_error:
#         return owner_error

#     payload = request.get_json(silent=True) or {}

#     resume_data = normalize_resume_data(payload.get("resume_data") or result.resume_data)
#     theme_color = validate_theme_color(payload.get("theme_color") or result.theme_color)
#     saved_template = validate_template(payload.get("saved_template") or result.saved_template)

#     try:
#         html = render_template(
#             f"cv/{saved_template}.html",
#             cv=resume_data,
#             template=saved_template,
#             theme=theme_color,
#         )
#         return jsonify({"ok": True, "html": html})
#     except Exception as e:
#         return jsonify({"ok": False, "error": f"Preview render failed: {str(e)}"}), 500


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port_raw = os.getenv("PORT", "5000")
    try:
        port = int(port_raw)
    except ValueError:
        port = 5000
    app.run(host=host, port=port, debug=IS_DEBUG)

# import os
# import re
# import time
# import uuid
# import json
# from pathlib import Path
# from urllib.parse import quote_plus

# from dotenv import load_dotenv
# from flask import (
#     Flask,
#     redirect,
#     render_template,
#     session,
#     url_for,
#     request,
#     jsonify,
# )
# from authlib.integrations.flask_client import OAuth
# from werkzeug.utils import secure_filename
# from google import genai
# from flask_sqlalchemy import SQLAlchemy


# basedir = os.path.abspath(os.path.dirname(__file__))
# load_dotenv(os.path.join(basedir, ".env"))

# app = Flask(__name__)
# app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-this-secret-key")

# app.config["SESSION_COOKIE_HTTPONLY"] = True
# app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
# app.config["SESSION_COOKIE_SECURE"] = False

# UPLOAD_FOLDER = os.path.join(basedir, "uploads")
# Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

# ALLOWED_EXTENSIONS = {"pdf", "doc", "docx", "txt", "html"}
# MAX_FILE_SIZE = 10 * 1024 * 1024
# MODEL_ID = "gemini-2.5-flash"

# ALLOWED_TEMPLATES = {
#     "classic",
#     "modern",
#     "sidebar",
#     "twocol",
#     "executive",
#     "creative",
# }
# DEFAULT_TEMPLATE = "classic"
# DEFAULT_THEME = "#ba7517"

# HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")

# google_client_id = os.getenv("GOOGLE_CLIENT_ID")
# google_client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
# google_api_key = os.getenv("GOOGLE_API_KEY")

# db_host = os.getenv("DB_HOST", "127.0.0.1")
# db_port = os.getenv("DB_PORT", "3306")
# db_name = os.getenv("DB_NAME", "test")
# db_user = os.getenv("DB_USER", "root")
# db_password = quote_plus(os.getenv("DB_PASSWORD", ""))

# if not google_client_id:
#     raise ValueError("GOOGLE_CLIENT_ID is missing in .env")
# if not google_client_secret:
#     raise ValueError("GOOGLE_CLIENT_SECRET is missing in .env")
# if not google_api_key:
#     raise ValueError("GOOGLE_API_KEY is missing in .env")

# app.config["SQLALCHEMY_DATABASE_URI"] = (
#     f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
# )
# app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# db = SQLAlchemy(app)
# oauth = OAuth(app)

# google = oauth.register(
#     name="google",
#     client_id=google_client_id,
#     client_secret=google_client_secret,
#     server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
#     client_kwargs={"scope": "openid email profile"},
# )

# gemini_client = genai.Client(api_key=google_api_key)


# class ResumeResult(db.Model):
#     __tablename__ = "resume_results"

#     id = db.Column(db.Integer, primary_key=True)
#     result_id = db.Column(db.String(64), unique=True, nullable=False, index=True)

#     user_google_id = db.Column(db.String(255), nullable=True)
#     user_name = db.Column(db.String(255), nullable=True)
#     user_email = db.Column(db.String(255), nullable=True, index=True)
#     user_picture = db.Column(db.Text, nullable=True)

#     original_filename = db.Column(db.String(255), nullable=False)
#     saved_path = db.Column(db.Text, nullable=False)

#     jd_text = db.Column(db.Text, nullable=True)
#     resume_data = db.Column(db.JSON, nullable=False)

#     theme_color = db.Column(db.String(20), nullable=False, default=DEFAULT_THEME)
#     saved_template = db.Column(db.String(32), nullable=True, default=DEFAULT_TEMPLATE)

#     created_at = db.Column(db.DateTime, server_default=db.func.now(), nullable=False)
#     updated_at = db.Column(
#         db.DateTime,
#         server_default=db.func.now(),
#         onupdate=db.func.now(),
#         nullable=False,
#     )


# @app.before_request
# def create_tables():
#     if not getattr(app, "_db_ready", False):
#         db.create_all()
#         app._db_ready = True


# def allowed_file(filename: str) -> bool:
#     return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# def wait_for_gemini_file(file_obj, timeout=60):
#     start = time.time()
#     while getattr(file_obj, "state", None) and file_obj.state.name == "PROCESSING":
#         if time.time() - start > timeout:
#             raise TimeoutError("Gemini file processing timed out.")
#         time.sleep(2)
#         file_obj = gemini_client.files.get(name=file_obj.name)
#     return file_obj


# def clean_json_response(text: str) -> str:
#     text = text.strip()
#     if text.startswith("```json"):
#         text = text[7:]
#     elif text.startswith("```"):
#         text = text[3:]
#     if text.endswith("```"):
#         text = text[:-3]
#     return text.strip()


# def ensure_list(value):
#     if isinstance(value, list):
#         return value
#     if value in (None, ""):
#         return []
#     return [str(value)]


# def normalize_resume_data(data):
#     if not isinstance(data, dict):
#         data = {}

#     normalized = {
#         "full_name": str(data.get("full_name", "") or ""),
#         "headline": str(data.get("headline", "") or ""),
#         "career_summary": str(data.get("career_summary", "") or ""),
#         "contact": {"phone": "", "email": "", "linkedin": "", "address": ""},
#         "work_experience": [],
#         "education": [],
#         "certifications": [],
#         "skills": {"software": [], "languages": [], "other": []},
#         "suggestions": [],
#     }

#     contact = data.get("contact")
#     if isinstance(contact, dict):
#         normalized["contact"]["phone"] = str(contact.get("phone", "") or "")
#         normalized["contact"]["email"] = str(contact.get("email", "") or "")
#         normalized["contact"]["linkedin"] = str(contact.get("linkedin", "") or "")
#         normalized["contact"]["address"] = str(contact.get("address", "") or "")

#     for job in ensure_list(data.get("work_experience")):
#         if not isinstance(job, dict):
#             continue
#         normalized["work_experience"].append(
#             {
#                 "job_title": str(job.get("job_title", "") or ""),
#                 "company": str(job.get("company", "") or ""),
#                 "location": str(job.get("location", "") or ""),
#                 "start_date": str(job.get("start_date", "") or ""),
#                 "end_date": str(job.get("end_date", "") or ""),
#                 "bullets": [str(x) for x in ensure_list(job.get("bullets")) if str(x).strip()],
#             }
#         )

#     for edu in ensure_list(data.get("education")):
#         if not isinstance(edu, dict):
#             continue
#         normalized["education"].append(
#             {
#                 "institution": str(edu.get("institution", "") or ""),
#                 "degree": str(edu.get("degree", "") or ""),
#                 "location": str(edu.get("location", "") or ""),
#                 "graduation_date": str(edu.get("graduation_date", "") or ""),
#                 "details": [str(x) for x in ensure_list(edu.get("details")) if str(x).strip()],
#             }
#         )

#     for cert in ensure_list(data.get("certifications")):
#         if not isinstance(cert, dict):
#             continue
#         normalized["certifications"].append(
#             {
#                 "name": str(cert.get("name", "") or ""),
#                 "issuer": str(cert.get("issuer", "") or ""),
#                 "date": str(cert.get("date", "") or ""),
#             }
#         )

#     skills = data.get("skills")
#     if isinstance(skills, dict):
#         normalized["skills"]["software"] = [
#             str(x) for x in ensure_list(skills.get("software")) if str(x).strip()
#         ]
#         normalized["skills"]["languages"] = [
#             str(x) for x in ensure_list(skills.get("languages")) if str(x).strip()
#         ]
#         normalized["skills"]["other"] = [
#             str(x) for x in ensure_list(skills.get("other")) if str(x).strip()
#         ]

#     normalized["suggestions"] = [
#         str(x) for x in ensure_list(data.get("suggestions")) if str(x).strip()
#     ]

#     return normalized


# def validate_theme_color(color: str) -> str:
#     color = str(color).strip().lower()
#     if color and not color.startswith("#"):
#         color = "#" + color
#     if HEX_COLOR_RE.match(color):
#         return color
#     return DEFAULT_THEME


# def validate_template(template: str) -> str:
#     if template in ALLOWED_TEMPLATES:
#         return template
#     return DEFAULT_TEMPLATE


# def get_result_or_404(result_id: str):
#     return ResumeResult.query.filter_by(result_id=result_id).first()


# def require_login_json():
#     user = session.get("user")
#     if not user:
#         return None, jsonify({"ok": False, "error": "Please log in first."}), 401
#     return user, None, None


# def require_login_page():
#     user = session.get("user")
#     if not user:
#         return None, redirect(url_for("login"))
#     return user, None


# def check_result_ownership(result: ResumeResult):
#     user = session.get("user")
#     if not user:
#         return jsonify({"ok": False, "error": "Please log in first."}), 401
#     if result.user_google_id and result.user_google_id != user.get("id"):
#         return jsonify({"ok": False, "error": "Forbidden."}), 403
#     return None


# def build_resume_view_context(result: ResumeResult, template_override=None, color_override=None):
#     template = validate_template(template_override or result.saved_template or DEFAULT_TEMPLATE)
#     theme = validate_theme_color(color_override or result.theme_color or DEFAULT_THEME)
#     cv = normalize_resume_data(result.resume_data)

#     return {
#         "user": session.get("user"),
#         "result": result,
#         "result_id": result.result_id,
#         "cv": cv,
#         "template": template,
#         "theme": theme,
#     }


# @app.route("/")
# def index():
#     return render_template("index.html", user=session.get("user"))


# @app.route("/login")
# def login():
#     redirect_uri = url_for("auth_callback", _external=True)
#     return google.authorize_redirect(redirect_uri)


# @app.route("/callback")
# def auth_callback():
#     token = google.authorize_access_token()
#     user_info = token.get("userinfo")
#     if not user_info:
#         user_info = google.get("userinfo").json()

#     session["user"] = {
#         "id": user_info.get("sub"),
#         "name": user_info.get("name"),
#         "email": user_info.get("email"),
#         "picture": user_info.get("picture"),
#     }

#     existing = ResumeResult.query.filter_by(
#         user_google_id=session["user"].get("id")
#     ).count()

#     if existing > 0:
#         return redirect(url_for("documents"))
#     return redirect(url_for("improve"))


# @app.route("/logout")
# def logout():
#     session.clear()
#     return redirect(url_for("index"))


# @app.route("/improve")
# def improve():
#     user, redirect_response = require_login_page()
#     if redirect_response:
#         return redirect_response
#     return render_template("improve.html", user=user)


# @app.route("/select-template/<result_id>")
# def select_template(result_id):
#     """
#     FIXED:
#     This now loads the ACTUAL generated CV from result.resume_data
#     and passes cv/template/theme to the template page.
#     """
#     result = get_result_or_404(result_id)
#     if not result:
#         return "Result not found", 404

#     template = validate_template(request.args.get("template", "").strip() or result.saved_template)
#     color_raw = request.args.get("color", "").strip()
#     theme = validate_theme_color(color_raw) if color_raw else validate_theme_color(result.theme_color)

#     changed = False
#     if result.saved_template != template:
#         result.saved_template = template
#         changed = True
#     if result.theme_color != theme:
#         result.theme_color = theme
#         changed = True
#     if changed:
#         db.session.commit()

#     context = build_resume_view_context(result)
#     return render_template("select_template.html", **context)


# @app.route("/edit/<result_id>")
# def edit_page(result_id):
#     user, redirect_response = require_login_page()
#     if redirect_response:
#         return redirect_response

#     result = get_result_or_404(result_id)
#     if not result:
#         return "Result not found", 404

#     template = validate_template(request.args.get("template", "").strip() or result.saved_template)
#     color_raw = request.args.get("color", "").strip()
#     theme = validate_theme_color(color_raw) if color_raw else validate_theme_color(result.theme_color)

#     changed = False
#     if result.saved_template != template:
#         result.saved_template = template
#         changed = True
#     if result.theme_color != theme:
#         result.theme_color = theme
#         changed = True
#     if changed:
#         db.session.commit()

#     context = build_resume_view_context(result)
#     return render_template("edit.html", **context)


# @app.route("/result/<result_id>")
# def result_page(result_id):
#     result = get_result_or_404(result_id)
#     if not result:
#         return "Result not found", 404

#     template = validate_template(request.args.get("template", "").strip() or result.saved_template)
#     color_raw = request.args.get("color", "").strip()
#     theme = validate_theme_color(color_raw) if color_raw else validate_theme_color(result.theme_color)

#     changed = False
#     if result.saved_template != template:
#         result.saved_template = template
#         changed = True
#     if result.theme_color != theme:
#         result.theme_color = theme
#         changed = True
#     if changed:
#         db.session.commit()

#     context = build_resume_view_context(result)
#     return render_template("result.html", **context)


# @app.route("/documents")
# def documents():
#     user, redirect_response = require_login_page()
#     if redirect_response:
#         return redirect_response

#     items = (
#         ResumeResult.query.filter_by(user_google_id=user.get("id"))
#         .order_by(ResumeResult.updated_at.desc())
#         .all()
#     )

#     return render_template("documents.html", user=user, items=items, resumes=items)


# @app.route("/api/result/<result_id>")
# def get_result_api(result_id):
#     result = get_result_or_404(result_id)
#     if not result:
#         return jsonify({"ok": False, "error": "Result not found."}), 404

#     owner_error = check_result_ownership(result)
#     if owner_error:
#         return owner_error

#     return jsonify(
#         {
#             "ok": True,
#             "result_id": result.result_id,
#             "resume_data": normalize_resume_data(result.resume_data),
#             "theme_color": validate_theme_color(result.theme_color),
#             "saved_template": validate_template(result.saved_template),
#         }
#     )


# @app.route("/api/result/<result_id>/update", methods=["POST"])
# def update_result_api(result_id):
#     result = get_result_or_404(result_id)
#     if not result:
#         return jsonify({"ok": False, "error": "Result not found."}), 404

#     owner_error = check_result_ownership(result)
#     if owner_error:
#         return owner_error

#     payload = request.get_json(silent=True) or {}

#     resume_data = normalize_resume_data(payload.get("resume_data") or result.resume_data)
#     theme_color = validate_theme_color(payload.get("theme_color") or result.theme_color)
#     saved_template = validate_template(payload.get("saved_template") or result.saved_template)

#     result.resume_data = resume_data
#     result.theme_color = theme_color
#     result.saved_template = saved_template
#     db.session.commit()

#     return jsonify({"ok": True})


# @app.route("/api/result/<result_id>/delete", methods=["DELETE"])
# def delete_result_api(result_id):
#     result = get_result_or_404(result_id)
#     if not result:
#         return jsonify({"ok": False, "error": "Result not found."}), 404

#     owner_error = check_result_ownership(result)
#     if owner_error:
#         return owner_error

#     db.session.delete(result)
#     db.session.commit()
#     return jsonify({"ok": True})


# @app.route("/api/user-cvs")
# def user_cvs_api():
#     user, error_response, status = require_login_json()
#     if error_response:
#         return error_response, status

#     items = (
#         ResumeResult.query.filter_by(user_google_id=user.get("id"))
#         .order_by(ResumeResult.updated_at.desc())
#         .limit(20)
#         .all()
#     )

#     return jsonify(
#         {
#             "ok": True,
#             "items": [
#                 {
#                     "result_id": item.result_id,
#                     "full_name": normalize_resume_data(item.resume_data).get("full_name", ""),
#                     "headline": normalize_resume_data(item.resume_data).get("headline", ""),
#                     "template": validate_template(item.saved_template),
#                     "theme_color": validate_theme_color(item.theme_color),
#                     "updated_at": item.updated_at.isoformat() if item.updated_at else None,
#                 }
#                 for item in items
#             ],
#         }
#     )


# @app.route("/api/reuse-cv", methods=["POST"])
# def reuse_cv_api():
#     user, error_response, status = require_login_json()
#     if error_response:
#         return error_response, status

#     payload = request.get_json(silent=True) or {}
#     reuse_id = str(payload.get("reuse_result_id", "")).strip()
#     jd_text = str(payload.get("jd_text", "")).strip()

#     if not reuse_id:
#         return jsonify({"ok": False, "error": "reuse_result_id is required."}), 400

#     existing = get_result_or_404(reuse_id)
#     if not existing:
#         return jsonify({"ok": False, "error": "Previous CV not found."}), 404

#     if existing.user_google_id and existing.user_google_id != user.get("id"):
#         return jsonify({"ok": False, "error": "Forbidden."}), 403

#     existing_json = json.dumps(normalize_resume_data(existing.resume_data), ensure_ascii=False)

#     prompt = f"""
# You are an expert ATS resume writer.

# Below is an existing CV in JSON format. Rewrite and improve it — strengthen the language,
# fix structure, and tailor it to the provided job description if available.

# IMPORTANT RULES:
# - Return ONLY valid JSON.
# - Do not wrap the JSON in markdown or triple backticks.
# - Keep wording concise and achievement-focused.
# - Preserve truthfulness. Do not invent experience.

# Existing CV JSON:
# {existing_json}

# Job Description:
# {jd_text if jd_text else "No job description provided."}

# Return JSON in exactly this structure:
# {{
#   "full_name": "",
#   "headline": "",
#   "contact": {{"phone": "", "email": "", "linkedin": "", "address": ""}},
#   "career_summary": "",
#   "work_experience": [{{"job_title": "", "company": "", "location": "", "start_date": "", "end_date": "", "bullets": [""]}}],
#   "education": [{{"institution": "", "degree": "", "location": "", "graduation_date": "", "details": [""]}}],
#   "certifications": [{{"name": "", "issuer": "", "date": ""}}],
#   "skills": {{"software": [""], "languages": [""], "other": [""]}},
#   "suggestions": [""]
# }}
# """

#     try:
#         response = gemini_client.models.generate_content(model=MODEL_ID, contents=[prompt])
#         raw_text = response.text or ""
#         cleaned_text = clean_json_response(raw_text)
#         resume_data = normalize_resume_data(json.loads(cleaned_text))

#         result_id = uuid.uuid4().hex
#         new_result = ResumeResult(
#             result_id=result_id,
#             user_google_id=user.get("id"),
#             user_name=user.get("name"),
#             user_email=user.get("email"),
#             user_picture=user.get("picture"),
#             original_filename=existing.original_filename,
#             saved_path="reused",
#             jd_text=jd_text,
#             resume_data=resume_data,
#             theme_color=existing.theme_color or DEFAULT_THEME,
#             saved_template=existing.saved_template or DEFAULT_TEMPLATE,
#         )
#         db.session.add(new_result)
#         db.session.commit()

#         return jsonify(
#             {
#                 "ok": True,
#                 "result_id": result_id,
#                 "redirect_url": url_for("select_template", result_id=result_id),
#             }
#         )
#     except json.JSONDecodeError:
#         return jsonify({"ok": False, "error": "Gemini returned invalid JSON. Please try again."}), 500
#     except Exception as e:
#         return jsonify({"ok": False, "error": str(e)}), 500


# @app.route("/api/improve-cv", methods=["POST"])
# def improve_cv_api():
#     user, error_response, status = require_login_json()
#     if error_response:
#         return error_response, status

#     jd_text = request.form.get("jd_text", "").strip()

#     # Reuse existing CV branch
#     reuse_id = request.form.get("reuse_result_id", "").strip()
#     if reuse_id:
#         existing = get_result_or_404(reuse_id)
#         if not existing:
#             return jsonify({"ok": False, "error": "Previous CV not found."}), 404
#         if existing.user_google_id and existing.user_google_id != user.get("id"):
#             return jsonify({"ok": False, "error": "Forbidden."}), 403

#         existing_json = json.dumps(normalize_resume_data(existing.resume_data), ensure_ascii=False)
#         prompt = f"""
# You are an expert ATS resume writer.

# Below is an existing CV in JSON format. Rewrite and improve it — strengthen the language,
# fix structure, and tailor it to the provided job description if available.

# IMPORTANT RULES:
# - Return ONLY valid JSON.
# - Do not wrap the JSON in markdown or triple backticks.
# - Keep wording concise and achievement-focused.
# - Preserve truthfulness. Do not invent experience.

# Existing CV JSON:
# {existing_json}

# Job Description:
# {jd_text if jd_text else "No job description provided."}

# Return JSON in exactly this structure:
# {{
#   "full_name": "",
#   "headline": "",
#   "contact": {{"phone": "", "email": "", "linkedin": "", "address": ""}},
#   "career_summary": "",
#   "work_experience": [{{"job_title": "", "company": "", "location": "", "start_date": "", "end_date": "", "bullets": [""]}}],
#   "education": [{{"institution": "", "degree": "", "location": "", "graduation_date": "", "details": [""]}}],
#   "certifications": [{{"name": "", "issuer": "", "date": ""}}],
#   "skills": {{"software": [""], "languages": [""], "other": [""]}},
#   "suggestions": [""]
# }}
# """
#         try:
#             response = gemini_client.models.generate_content(model=MODEL_ID, contents=[prompt])
#             raw_text = response.text or ""
#             cleaned_text = clean_json_response(raw_text)
#             resume_data = normalize_resume_data(json.loads(cleaned_text))

#             result_id = uuid.uuid4().hex
#             new_result = ResumeResult(
#                 result_id=result_id,
#                 user_google_id=user.get("id"),
#                 user_name=user.get("name"),
#                 user_email=user.get("email"),
#                 user_picture=user.get("picture"),
#                 original_filename=existing.original_filename,
#                 saved_path="reused",
#                 jd_text=jd_text,
#                 resume_data=resume_data,
#                 theme_color=existing.theme_color or DEFAULT_THEME,
#                 saved_template=existing.saved_template or DEFAULT_TEMPLATE,
#             )
#             db.session.add(new_result)
#             db.session.commit()

#             return jsonify(
#                 {
#                     "ok": True,
#                     "result_id": result_id,
#                     "redirect_url": url_for("select_template", result_id=result_id),
#                 }
#             )
#         except json.JSONDecodeError:
#             return jsonify({"ok": False, "error": "Gemini returned invalid JSON. Please try again."}), 500
#         except Exception as e:
#             return jsonify({"ok": False, "error": str(e)}), 500

#     if "cv_file" not in request.files:
#         return jsonify({"ok": False, "error": "CV file is required."}), 400

#     cv_file = request.files["cv_file"]

#     if cv_file.filename == "":
#         return jsonify({"ok": False, "error": "No file selected."}), 400

#     if not allowed_file(cv_file.filename):
#         return jsonify({"ok": False, "error": "Unsupported CV file type."}), 400

#     cv_file.seek(0, os.SEEK_END)
#     size = cv_file.tell()
#     cv_file.seek(0)

#     if size > MAX_FILE_SIZE:
#         return jsonify({"ok": False, "error": "CV file must be under 10 MB."}), 400

#     safe_name = secure_filename(cv_file.filename)
#     unique_name = f"{uuid.uuid4().hex}_{safe_name}"
#     saved_path = os.path.join(UPLOAD_FOLDER, unique_name)
#     cv_file.save(saved_path)

#     try:
#         uploaded_cv = gemini_client.files.upload(file=saved_path)
#         uploaded_cv = wait_for_gemini_file(uploaded_cv)

#         if not getattr(uploaded_cv, "state", None) or uploaded_cv.state.name != "ACTIVE":
#             state_name = getattr(getattr(uploaded_cv, "state", None), "name", "UNKNOWN")
#             return jsonify({"ok": False, "error": f"Gemini file processing failed: {state_name}"}), 500

#         prompt = f"""
# You are an expert ATS resume writer.

# Read the uploaded CV and the user details. Rewrite the resume into a clean, ATS-friendly format.

# IMPORTANT RULES:
# - Return ONLY valid JSON.
# - Do not wrap the JSON in markdown or triple backticks.
# - Do not include explanations outside JSON.
# - Keep wording concise and achievement-focused.
# - Use strong action verbs.
# - Tailor the content to the provided job description if available.
# - Preserve truthfulness. Do not invent experience.

# User details:
# - Name: {user.get("name", "")}
# - Email: {user.get("email", "")}

# Job Description:
# {jd_text if jd_text else "No job description provided."}

# Return JSON in exactly this structure:
# {{
#   "full_name": "",
#   "headline": "",
#   "contact": {{
#     "phone": "",
#     "email": "",
#     "linkedin": "",
#     "address": ""
#   }},
#   "career_summary": "",
#   "work_experience": [
#     {{
#       "job_title": "",
#       "company": "",
#       "location": "",
#       "start_date": "",
#       "end_date": "",
#       "bullets": [""]
#     }}
#   ],
#   "education": [
#     {{
#       "institution": "",
#       "degree": "",
#       "location": "",
#       "graduation_date": "",
#       "details": [""]
#     }}
#   ],
#   "certifications": [
#     {{
#       "name": "",
#       "issuer": "",
#       "date": ""
#     }}
#   ],
#   "skills": {{
#     "software": [""],
#     "languages": [""],
#     "other": [""]
#   }},
#   "suggestions": [""]
# }}
# """

#         response = gemini_client.models.generate_content(
#             model=MODEL_ID,
#             contents=[uploaded_cv, prompt],
#         )

#         raw_text = response.text or ""
#         cleaned_text = clean_json_response(raw_text)
#         resume_data = normalize_resume_data(json.loads(cleaned_text))

#         result_id = uuid.uuid4().hex

#         new_result = ResumeResult(
#             result_id=result_id,
#             user_google_id=user.get("id"),
#             user_name=user.get("name"),
#             user_email=user.get("email"),
#             user_picture=user.get("picture"),
#             original_filename=cv_file.filename,
#             saved_path=saved_path,
#             jd_text=jd_text,
#             resume_data=resume_data,
#             theme_color=DEFAULT_THEME,
#             saved_template=DEFAULT_TEMPLATE,
#         )

#         db.session.add(new_result)
#         db.session.commit()

#         try:
#             os.remove(saved_path)
#         except OSError:
#             pass

#         return jsonify(
#             {
#                 "ok": True,
#                 "result_id": result_id,
#                 "redirect_url": url_for("select_template", result_id=result_id),
#             }
#         )

#     except json.JSONDecodeError:
#         return jsonify({"ok": False, "error": "Gemini returned invalid JSON. Please try again."}), 500
#     except Exception as e:
#         return jsonify({"ok": False, "error": str(e)}), 500


# if __name__ == "__main__":
#     app.run(debug=True)

# # import os
# # import re
# # import time
# # import uuid
# # import json
# # from pathlib import Path
# # from urllib.parse import quote_plus

# # from dotenv import load_dotenv
# # from flask import (
# #     Flask,
# #     redirect,
# #     render_template,
# #     session,
# #     url_for,
# #     request,
# #     jsonify,
# # )
# # from authlib.integrations.flask_client import OAuth
# # from werkzeug.utils import secure_filename
# # from google import genai
# # from flask_sqlalchemy import SQLAlchemy

# # basedir = os.path.abspath(os.path.dirname(__file__))
# # load_dotenv(os.path.join(basedir, ".env"))

# # app = Flask(__name__)
# # app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-this-secret-key")

# # app.config["SESSION_COOKIE_HTTPONLY"] = True
# # app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
# # app.config["SESSION_COOKIE_SECURE"] = False

# # UPLOAD_FOLDER = os.path.join(basedir, "uploads")
# # Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

# # ALLOWED_EXTENSIONS = {"pdf", "doc", "docx", "txt", "html"}
# # MAX_FILE_SIZE = 10 * 1024 * 1024
# # MODEL_ID = "gemini-2.5-flash"

# # ALLOWED_TEMPLATES = {"classic", "modern", "sidebar", "twocol", "executive", "creative"}
# # HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
# # DEFAULT_TEMPLATE = "classic"
# # DEFAULT_THEME = "#ba7517"

# # google_client_id = os.getenv("GOOGLE_CLIENT_ID")
# # google_client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
# # google_api_key = os.getenv("GOOGLE_API_KEY")

# # db_host = os.getenv("DB_HOST", "127.0.0.1")
# # db_port = os.getenv("DB_PORT", "3306")
# # db_name = os.getenv("DB_NAME", "test")
# # db_user = os.getenv("DB_USER", "root")
# # db_password = quote_plus(os.getenv("DB_PASSWORD", ""))

# # if not google_client_id:
# #     raise ValueError("GOOGLE_CLIENT_ID is missing in .env")
# # if not google_client_secret:
# #     raise ValueError("GOOGLE_CLIENT_SECRET is missing in .env")
# # if not google_api_key:
# #     raise ValueError("GOOGLE_API_KEY is missing in .env")

# # app.config["SQLALCHEMY_DATABASE_URI"] = (
# #     f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
# # )
# # app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# # db = SQLAlchemy(app)
# # oauth = OAuth(app)

# # google = oauth.register(
# #     name="google",
# #     client_id=google_client_id,
# #     client_secret=google_client_secret,
# #     server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
# #     client_kwargs={"scope": "openid email profile"},
# # )

# # gemini_client = genai.Client(api_key=google_api_key)


# # class ResumeResult(db.Model):
# #     __tablename__ = "resume_results"

# #     id = db.Column(db.Integer, primary_key=True)
# #     result_id = db.Column(db.String(64), unique=True, nullable=False, index=True)

# #     user_google_id = db.Column(db.String(255), nullable=True)
# #     user_name = db.Column(db.String(255), nullable=True)
# #     user_email = db.Column(db.String(255), nullable=True, index=True)
# #     user_picture = db.Column(db.Text, nullable=True)

# #     original_filename = db.Column(db.String(255), nullable=False)
# #     saved_path = db.Column(db.Text, nullable=False)

# #     jd_text = db.Column(db.Text, nullable=True)
# #     resume_data = db.Column(db.JSON, nullable=False)

# #     theme_color = db.Column(db.String(20), nullable=False, default=DEFAULT_THEME)
# #     saved_template = db.Column(db.String(32), nullable=True, default=DEFAULT_TEMPLATE)

# #     created_at = db.Column(db.DateTime, server_default=db.func.now(), nullable=False)
# #     updated_at = db.Column(db.DateTime, server_default=db.func.now(), onupdate=db.func.now(), nullable=False)


# # @app.before_request
# # def create_tables():
# #     if not getattr(app, "_db_ready", False):
# #         db.create_all()
# #         app._db_ready = True


# # def allowed_file(filename: str) -> bool:
# #     return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# # def wait_for_gemini_file(file_obj, timeout=60):
# #     start = time.time()
# #     while getattr(file_obj, "state", None) and file_obj.state.name == "PROCESSING":
# #         if time.time() - start > timeout:
# #             raise TimeoutError("Gemini file processing timed out.")
# #         time.sleep(2)
# #         file_obj = gemini_client.files.get(name=file_obj.name)
# #     return file_obj


# # def clean_json_response(text: str) -> str:
# #     text = text.strip()
# #     if text.startswith("```json"):
# #         text = text[7:]
# #     elif text.startswith("```"):
# #         text = text[3:]
# #     if text.endswith("```"):
# #         text = text[:-3]
# #     return text.strip()


# # def ensure_list(value):
# #     if isinstance(value, list):
# #         return value
# #     if value in (None, ""):
# #         return []
# #     return [str(value)]


# # def normalize_resume_data(data):
# #     if not isinstance(data, dict):
# #         data = {}

# #     normalized = {
# #         "full_name": str(data.get("full_name", "") or ""),
# #         "headline": str(data.get("headline", "") or ""),
# #         "career_summary": str(data.get("career_summary", "") or ""),
# #         "contact": {"phone": "", "email": "", "linkedin": "", "address": ""},
# #         "work_experience": [],
# #         "education": [],
# #         "certifications": [],
# #         "skills": {"software": [], "languages": [], "other": []},
# #         "suggestions": [],
# #     }

# #     contact = data.get("contact")
# #     if isinstance(contact, dict):
# #         normalized["contact"]["phone"]    = str(contact.get("phone", "") or "")
# #         normalized["contact"]["email"]    = str(contact.get("email", "") or "")
# #         normalized["contact"]["linkedin"] = str(contact.get("linkedin", "") or "")
# #         normalized["contact"]["address"]  = str(contact.get("address", "") or "")

# #     for job in ensure_list(data.get("work_experience")):
# #         if not isinstance(job, dict):
# #             continue
# #         normalized["work_experience"].append({
# #             "job_title":  str(job.get("job_title", "") or ""),
# #             "company":    str(job.get("company", "") or ""),
# #             "location":   str(job.get("location", "") or ""),
# #             "start_date": str(job.get("start_date", "") or ""),
# #             "end_date":   str(job.get("end_date", "") or ""),
# #             "bullets":    [str(x) for x in ensure_list(job.get("bullets")) if str(x).strip()],
# #         })

# #     for edu in ensure_list(data.get("education")):
# #         if not isinstance(edu, dict):
# #             continue
# #         normalized["education"].append({
# #             "institution":    str(edu.get("institution", "") or ""),
# #             "degree":         str(edu.get("degree", "") or ""),
# #             "location":       str(edu.get("location", "") or ""),
# #             "graduation_date": str(edu.get("graduation_date", "") or ""),
# #             "details":        [str(x) for x in ensure_list(edu.get("details")) if str(x).strip()],
# #         })

# #     for cert in ensure_list(data.get("certifications")):
# #         if not isinstance(cert, dict):
# #             continue
# #         normalized["certifications"].append({
# #             "name":   str(cert.get("name", "") or ""),
# #             "issuer": str(cert.get("issuer", "") or ""),
# #             "date":   str(cert.get("date", "") or ""),
# #         })

# #     skills = data.get("skills")
# #     if isinstance(skills, dict):
# #         normalized["skills"]["software"]  = [str(x) for x in ensure_list(skills.get("software")) if str(x).strip()]
# #         normalized["skills"]["languages"] = [str(x) for x in ensure_list(skills.get("languages")) if str(x).strip()]
# #         normalized["skills"]["other"]     = [str(x) for x in ensure_list(skills.get("other")) if str(x).strip()]

# #     normalized["suggestions"] = [str(x) for x in ensure_list(data.get("suggestions")) if str(x).strip()]

# #     return normalized


# # def validate_theme_color(color: str) -> str:
# #     color = str(color).strip().lower()
# #     if color and not color.startswith("#"):
# #         color = "#" + color
# #     if HEX_COLOR_RE.match(color):
# #         return color
# #     return DEFAULT_THEME


# # def validate_template(template: str) -> str:
# #     if template in ALLOWED_TEMPLATES:
# #         return template
# #     return DEFAULT_TEMPLATE


# # def get_resume_or_404(result_id: str):
# #     return ResumeResult.query.filter_by(result_id=result_id).first()


# # def enforce_owner_or_403(result: ResumeResult):
# #     user = session.get("user")
# #     if not user:
# #         return jsonify({"ok": False, "error": "Please log in first."}), 401
# #     if result.user_google_id and result.user_google_id != user.get("id"):
# #         return jsonify({"ok": False, "error": "Forbidden."}), 403
# #     return None


# # def build_cv_context(result: ResumeResult):
# #     template = validate_template(result.saved_template or DEFAULT_TEMPLATE)
# #     theme = validate_theme_color(result.theme_color or DEFAULT_THEME)
# #     cv = normalize_resume_data(result.resume_data)
# #     return {
# #         "cv": cv,
# #         "template": template,
# #         "theme": theme,
# #         "result": result,
# #         "result_id": result.result_id,
# #         "user": session.get("user"),
# #     }


# # # ─────────────────────────────────────────
# # #  Routes
# # # ─────────────────────────────────────────

# # @app.route("/")
# # def index():
# #     return render_template("index.html", user=session.get("user"))


# # @app.route("/improve")
# # def improve():
# #     user = session.get("user")
# #     if not user:
# #         return redirect(url_for("login"))
# #     return render_template("improve.html", user=user)


# # # Template selection page — shown right after the API returns
# # @app.route("/select-template/<result_id>")
# # def select_template(result_id):
# #     result = get_resume_or_404(result_id)
# #     if not result:
# #         return "Result not found", 404

# #     context = build_cv_context(result)
# #     return render_template("select_template.html", **context)


# # # ─────────────────────────────────────────
# # #  NEW: Step-by-step guided CV editor
# # #  Flow: select-template → edit → result
# # # ─────────────────────────────────────────
# # @app.route("/edit/<result_id>")
# # def edit_page(result_id):
# #     user = session.get("user")
# #     if not user:
# #         return redirect(url_for("login"))

# #     result = get_resume_or_404(result_id)
# #     if not result:
# #         return "Result not found", 404

# #     template = validate_template(
# #         request.args.get("template", "").strip() or result.saved_template or DEFAULT_TEMPLATE
# #     )
# #     theme = validate_theme_color(
# #         request.args.get("color", "").strip() or result.theme_color or DEFAULT_THEME
# #     )

# #     changed = False
# #     if result.saved_template != template:
# #         result.saved_template = template
# #         changed = True
# #     if result.theme_color != theme:
# #         result.theme_color = theme
# #         changed = True
# #     if changed:
# #         db.session.commit()

# #     context = build_cv_context(result)
# #     return render_template("edit.html", **context)


# # # Result page — renders the chosen template
# # @app.route("/result/<result_id>")
# # def result_page(result_id):
# #     result = get_resume_or_404(result_id)
# #     if not result:
# #         return "Result not found", 404

# #     template = validate_template(
# #         request.args.get("template", "").strip() or result.saved_template or DEFAULT_TEMPLATE
# #     )
# #     theme = validate_theme_color(
# #         request.args.get("color", "").strip() or result.theme_color or DEFAULT_THEME
# #     )

# #     changed = False
# #     if result.saved_template != template:
# #         result.saved_template = template
# #         changed = True
# #     if result.theme_color != theme:
# #         result.theme_color = theme
# #         changed = True
# #     if changed:
# #         db.session.commit()

# #     context = build_cv_context(result)
# #     return render_template("result.html", **context)


# # @app.route("/api/improve-cv", methods=["POST"])
# # def improve_cv_api():
# #     user = session.get("user")
# #     if not user:
# #         return jsonify({"ok": False, "error": "Please log in first."}), 401

# #     jd_text = request.form.get("jd_text", "").strip()

# #     # ── Reuse an existing CV result ──────────────────────────────────
# #     reuse_id = request.form.get("reuse_result_id", "").strip()
# #     if reuse_id:
# #         existing = ResumeResult.query.filter_by(result_id=reuse_id).first()
# #         if not existing:
# #             return jsonify({"ok": False, "error": "Previous CV not found."}), 404
# #         if existing.user_google_id and existing.user_google_id != user.get("id"):
# #             return jsonify({"ok": False, "error": "Forbidden."}), 403

# #         # Re-run Gemini on the saved resume_data JSON (as text prompt)
# #         import json as _json
# #         existing_json = _json.dumps(normalize_resume_data(existing.resume_data), ensure_ascii=False)
# #         prompt = f"""
# # You are an expert ATS resume writer.

# # Below is an existing CV in JSON format. Rewrite and improve it — strengthen the language,
# # fix structure, and tailor it to the provided job description if available.

# # IMPORTANT RULES:
# # - Return ONLY valid JSON.
# # - Do not wrap the JSON in markdown or triple backticks.
# # - Keep wording concise and achievement-focused.
# # - Preserve truthfulness. Do not invent experience.

# # Existing CV JSON:
# # {existing_json}

# # Job Description:
# # {jd_text if jd_text else "No job description provided."}

# # Return JSON in exactly this structure:
# # {{
# #   "full_name": "",
# #   "headline": "",
# #   "contact": {{"phone": "", "email": "", "linkedin": "", "address": ""}},
# #   "career_summary": "",
# #   "work_experience": [{{"job_title": "", "company": "", "location": "", "start_date": "", "end_date": "", "bullets": [""]}}],
# #   "education": [{{"institution": "", "degree": "", "location": "", "graduation_date": "", "details": [""]}}],
# #   "certifications": [{{"name": "", "issuer": "", "date": ""}}],
# #   "skills": {{"software": [""], "languages": [""], "other": [""]}},
# #   "suggestions": [""]
# # }}
# # """
# #         try:
# #             response = gemini_client.models.generate_content(model=MODEL_ID, contents=[prompt])
# #             raw_text = response.text or ""
# #             cleaned_text = clean_json_response(raw_text)
# #             resume_data = _json.loads(cleaned_text)
# #             resume_data = normalize_resume_data(resume_data)

# #             result_id = uuid.uuid4().hex
# #             new_result = ResumeResult(
# #                 result_id=result_id,
# #                 user_google_id=user.get("id"),
# #                 user_name=user.get("name"),
# #                 user_email=user.get("email"),
# #                 user_picture=user.get("picture"),
# #                 original_filename=existing.original_filename,
# #                 saved_path="reused",
# #                 jd_text=jd_text,
# #                 resume_data=resume_data,
# #                 theme_color=existing.theme_color or "#ba7517",
# #                 saved_template=existing.saved_template or "classic",
# #             )
# #             db.session.add(new_result)
# #             db.session.commit()

# #             return jsonify({
# #                 "ok": True,
# #                 "result_id": result_id,
# #                 "redirect_url": url_for("select_template", result_id=result_id),
# #             })
# #         except json.JSONDecodeError:
# #             return jsonify({"ok": False, "error": "Gemini returned invalid JSON. Please try again."}), 500
# #         except Exception as e:
# #             return jsonify({"ok": False, "error": str(e)}), 500
# #     # ── End reuse branch ─────────────────────────────────────────────

# #     if "cv_file" not in request.files:
# #         return jsonify({"ok": False, "error": "CV file is required."}), 400

# #     cv_file = request.files["cv_file"]

# #     if cv_file.filename == "":
# #         return jsonify({"ok": False, "error": "No file selected."}), 400

# #     if not allowed_file(cv_file.filename):
# #         return jsonify({"ok": False, "error": "Unsupported CV file type."}), 400

# #     cv_file.seek(0, os.SEEK_END)
# #     size = cv_file.tell()
# #     cv_file.seek(0)

# #     if size > MAX_FILE_SIZE:
# #         return jsonify({"ok": False, "error": "CV file must be under 10 MB."}), 400

# #     safe_name = secure_filename(cv_file.filename)
# #     unique_name = f"{uuid.uuid4().hex}_{safe_name}"
# #     saved_path = os.path.join(UPLOAD_FOLDER, unique_name)
# #     cv_file.save(saved_path)

# #     try:
# #         uploaded_cv = gemini_client.files.upload(file=saved_path)
# #         uploaded_cv = wait_for_gemini_file(uploaded_cv)

# #         if not getattr(uploaded_cv, "state", None) or uploaded_cv.state.name != "ACTIVE":
# #             state_name = getattr(getattr(uploaded_cv, "state", None), "name", "UNKNOWN")
# #             return jsonify({"ok": False, "error": f"Gemini file processing failed: {state_name}"}), 500

# #         prompt = f"""
# # You are an expert ATS resume writer.

# # Read the uploaded CV and the user details. Rewrite the resume into a clean, ATS-friendly format.

# # IMPORTANT RULES:
# # - Return ONLY valid JSON.
# # - Do not wrap the JSON in markdown or triple backticks.
# # - Do not include explanations outside JSON.
# # - Keep wording concise and achievement-focused.
# # - Use strong action verbs.
# # - Tailor the content to the provided job description if available.
# # - Preserve truthfulness. Do not invent experience.

# # User details:
# # - Name: {user.get("name", "")}
# # - Email: {user.get("email", "")}

# # Job Description:
# # {jd_text if jd_text else "No job description provided."}

# # Return JSON in exactly this structure:
# # {{
# #   "full_name": "",
# #   "headline": "",
# #   "contact": {{
# #     "phone": "",
# #     "email": "",
# #     "linkedin": "",
# #     "address": ""
# #   }},
# #   "career_summary": "",
# #   "work_experience": [
# #     {{
# #       "job_title": "",
# #       "company": "",
# #       "location": "",
# #       "start_date": "",
# #       "end_date": "",
# #       "bullets": [""]
# #     }}
# #   ],
# #   "education": [
# #     {{
# #       "institution": "",
# #       "degree": "",
# #       "location": "",
# #       "graduation_date": "",
# #       "details": [""]
# #     }}
# #   ],
# #   "certifications": [
# #     {{
# #       "name": "",
# #       "issuer": "",
# #       "date": ""
# #     }}
# #   ],
# #   "skills": {{
# #     "software": [""],
# #     "languages": [""],
# #     "other": [""]
# #   }},
# #   "suggestions": [""]
# # }}
# # """

# #         response = gemini_client.models.generate_content(
# #             model=MODEL_ID,
# #             contents=[uploaded_cv, prompt],
# #         )

# #         raw_text = response.text or ""
# #         cleaned_text = clean_json_response(raw_text)
# #         resume_data = json.loads(cleaned_text)
# #         resume_data = normalize_resume_data(resume_data)

# #         result_id = uuid.uuid4().hex

# #         new_result = ResumeResult(
# #             result_id=result_id,
# #             user_google_id=user.get("id"),
# #             user_name=user.get("name"),
# #             user_email=user.get("email"),
# #             user_picture=user.get("picture"),
# #             original_filename=cv_file.filename,
# #             saved_path=saved_path,
# #             jd_text=jd_text,
# #             resume_data=resume_data,
# #             theme_color="#ba7517",
# #         )

# #         db.session.add(new_result)
# #         db.session.commit()

# #         # Clean up local upload after Gemini has processed it
# #         try:
# #             os.remove(saved_path)
# #         except OSError:
# #             pass

# #         # Redirect to template picker
# #         return jsonify({
# #             "ok": True,
# #             "message": "CV processed successfully.",
# #             "result_id": result_id,
# #             "redirect_url": url_for("select_template", result_id=result_id),
# #         })

# #     except json.JSONDecodeError:
# #         return jsonify({"ok": False, "error": "Gemini returned invalid JSON. Please try again."}), 500
# #     except Exception as e:
# #         return jsonify({"ok": False, "error": str(e)}), 500


# # @app.route("/api/result/<result_id>", methods=["GET"])
# # def get_result_data(result_id):
# #     result = get_resume_or_404(result_id)
# #     if not result:
# #         return jsonify({"ok": False, "error": "Result not found."}), 404

# #     owner_check = enforce_owner_or_403(result)
# #     if owner_check:
# #         return owner_check

# #     return jsonify({
# #         "ok": True,
# #         "result_id": result.result_id,
# #         "resume_data": normalize_resume_data(result.resume_data),
# #         "theme_color": validate_theme_color(result.theme_color),
# #         "saved_template": validate_template(result.saved_template),
# #     })


# # @app.route("/api/user-cvs", methods=["GET"])
# # def user_cvs():
# #     user = session.get("user")
# #     if not user:
# #         return jsonify({"ok": False, "error": "Please log in first."}), 401

# #     resumes = ResumeResult.query.filter_by(
# #         user_google_id=user.get("id")
# #     ).order_by(ResumeResult.updated_at.desc()).limit(20).all()

# #     items = [
# #         {
# #             "result_id": r.result_id,
# #             "full_name": (r.resume_data or {}).get("full_name", "") or r.user_name or "Untitled CV",
# #             "headline": (r.resume_data or {}).get("headline", ""),
# #             "template": validate_template(r.saved_template),
# #             "theme_color": validate_theme_color(r.theme_color),
# #             "updated_at": r.updated_at.isoformat() if r.updated_at else None,
# #             "original_filename": r.original_filename or "",
# #         }
# #         for r in resumes
# #     ]

# #     return jsonify({
# #         "ok": True,
# #         "items": items,
# #         "cvs": items,
# #     })


# # @app.route("/api/reuse-cv", methods=["POST"])
# # def reuse_cv_api():
# #     user = session.get("user")
# #     if not user:
# #         return jsonify({"ok": False, "error": "Please log in first."}), 401

# #     data = request.get_json(silent=True) or {}
# #     result_id = str(data.get("reuse_result_id", "") or data.get("result_id", "")).strip()
# #     jd_text = str(data.get("jd_text", "")).strip()

# #     if not result_id:
# #         return jsonify({"ok": False, "error": "reuse_result_id is required."}), 400

# #     result = get_resume_or_404(result_id)
# #     if not result:
# #         return jsonify({"ok": False, "error": "Previous CV not found."}), 404

# #     if result.user_google_id and result.user_google_id != user.get("id"):
# #         return jsonify({"ok": False, "error": "Forbidden."}), 403

# #     existing_json = json.dumps(normalize_resume_data(result.resume_data), ensure_ascii=False)

# #     prompt = f"""
# # You are an expert ATS resume writer.

# # Below is an existing CV in JSON format. Rewrite and improve it - strengthen the language,
# # fix structure, and tailor it to the provided job description if available.

# # IMPORTANT RULES:
# # - Return ONLY valid JSON.
# # - Do not wrap the JSON in markdown or triple backticks.
# # - Keep wording concise and achievement-focused.
# # - Preserve truthfulness. Do not invent experience.

# # Existing CV JSON:
# # {existing_json}

# # Job Description:
# # {jd_text if jd_text else "No job description provided."}

# # Return JSON in exactly this structure:
# # {{
# #   "full_name": "",
# #   "headline": "",
# #   "contact": {{"phone": "", "email": "", "linkedin": "", "address": ""}},
# #   "career_summary": "",
# #   "work_experience": [{{"job_title": "", "company": "", "location": "", "start_date": "", "end_date": "", "bullets": [""]}}],
# #   "education": [{{"institution": "", "degree": "", "location": "", "graduation_date": "", "details": [""]}}],
# #   "certifications": [{{"name": "", "issuer": "", "date": ""}}],
# #   "skills": {{"software": [""], "languages": [""], "other": [""]}},
# #   "suggestions": [""]
# # }}
# # """

# #     try:
# #         response = gemini_client.models.generate_content(
# #             model=MODEL_ID,
# #             contents=[prompt],
# #         )
# #         raw_text = response.text or ""
# #         cleaned_text = clean_json_response(raw_text)
# #         resume_data = json.loads(cleaned_text)
# #         resume_data = normalize_resume_data(resume_data)

# #         new_id = uuid.uuid4().hex
# #         new_result = ResumeResult(
# #             result_id=new_id,
# #             user_google_id=user.get("id"),
# #             user_name=user.get("name"),
# #             user_email=user.get("email"),
# #             user_picture=user.get("picture"),
# #             original_filename=result.original_filename or "reused_cv",
# #             saved_path="reused",
# #             jd_text=jd_text,
# #             resume_data=resume_data,
# #             theme_color=result.theme_color or DEFAULT_THEME,
# #             saved_template=result.saved_template or DEFAULT_TEMPLATE,
# #         )
# #         db.session.add(new_result)
# #         db.session.commit()

# #         return jsonify({
# #             "ok": True,
# #             "result_id": new_id,
# #             "redirect_url": url_for("select_template", result_id=new_id),
# #         })

# #     except json.JSONDecodeError:
# #         return jsonify({"ok": False, "error": "Gemini returned invalid JSON. Please try again."}), 500
# #     except Exception as e:
# #         return jsonify({"ok": False, "error": str(e)}), 500


# # @app.route("/api/result/<result_id>/update", methods=["POST"])
# # def update_result_data(result_id):
# #     result = get_resume_or_404(result_id)
# #     if not result:
# #         return jsonify({"ok": False, "error": "Result not found."}), 404

# #     owner_check = enforce_owner_or_403(result)
# #     if owner_check:
# #         return owner_check

# #     data = request.get_json(silent=True) or {}
# #     result.resume_data = normalize_resume_data(data.get("resume_data") or {})
# #     result.theme_color = validate_theme_color(data.get("theme_color") or result.theme_color)
# #     result.saved_template = validate_template(data.get("saved_template") or result.saved_template)

# #     db.session.commit()
# #     return jsonify({"ok": True})


# # @app.route("/login")
# # def login():
# #     redirect_uri = url_for("callback", _external=True)
# #     return google.authorize_redirect(redirect_uri)


# # @app.route("/callback")
# # def callback():
# #     token = google.authorize_access_token()
# #     user_info = token.get("userinfo")
# #     if not user_info:
# #         user_info = google.parse_id_token(token)

# #     session["user"] = {
# #         "id":      user_info.get("sub"),
# #         "name":    user_info.get("name"),
# #         "email":   user_info.get("email"),
# #         "picture": user_info.get("picture"),
# #     }
# #     # If user has existing CVs, go to documents page; else improve
# #     existing = ResumeResult.query.filter_by(user_google_id=user_info.get("sub")).first()
# #     if existing:
# #         return redirect(url_for("documents"))
# #     return redirect(url_for("improve"))


# # @app.route("/logout")
# # def logout():
# #     session.clear()
# #     return redirect(url_for("index"))


# # # ─────────────────────────────────────────
# # #  Documents page — user's saved CVs
# # # ─────────────────────────────────────────
# # @app.route("/documents")
# # def documents():
# #     user = session.get("user")
# #     if not user:
# #         return redirect(url_for("login"))

# #     resumes = ResumeResult.query.filter_by(
# #         user_google_id=user.get("id")
# #     ).order_by(ResumeResult.updated_at.desc()).all()

# #     return render_template("documents.html", user=user, resumes=resumes, items=resumes)


# # # ─────────────────────────────────────────
# # #  Delete a CV
# # # ─────────────────────────────────────────
# # @app.route("/api/result/<result_id>/delete", methods=["DELETE"])
# # def delete_result(result_id):
# #     result = get_resume_or_404(result_id)
# #     if not result:
# #         return jsonify({"ok": False, "error": "Result not found."}), 404

# #     owner_check = enforce_owner_or_403(result)
# #     if owner_check:
# #         return owner_check

# #     db.session.delete(result)
# #     db.session.commit()
# #     return jsonify({"ok": True})


# # if __name__ == "__main__":
# #     app.run(debug=True)

# # # import os
# # # import re
# # # import time
# # # import uuid
# # # import json
# # # from pathlib import Path
# # # from urllib.parse import quote_plus

# # # from dotenv import load_dotenv
# # # from flask import (
# # #     Flask,
# # #     redirect,
# # #     render_template,
# # #     session,
# # #     url_for,
# # #     request,
# # #     jsonify,
# # # )
# # # from authlib.integrations.flask_client import OAuth
# # # from werkzeug.utils import secure_filename
# # # from google import genai
# # # from flask_sqlalchemy import SQLAlchemy

# # # basedir = os.path.abspath(os.path.dirname(__file__))
# # # load_dotenv(os.path.join(basedir, ".env"))

# # # app = Flask(__name__)
# # # app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-this-secret-key")

# # # app.config["SESSION_COOKIE_HTTPONLY"] = True
# # # app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
# # # app.config["SESSION_COOKIE_SECURE"] = False

# # # UPLOAD_FOLDER = os.path.join(basedir, "uploads")
# # # Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

# # # ALLOWED_EXTENSIONS = {"pdf", "doc", "docx", "txt", "html"}
# # # MAX_FILE_SIZE = 10 * 1024 * 1024
# # # MODEL_ID = "gemini-2.5-flash"

# # # ALLOWED_TEMPLATES = {"classic", "modern", "sidebar", "twocol", "executive", "creative"}
# # # HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")

# # # google_client_id = os.getenv("GOOGLE_CLIENT_ID")
# # # google_client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
# # # google_api_key = os.getenv("GOOGLE_API_KEY")

# # # db_host = os.getenv("DB_HOST", "127.0.0.1")
# # # db_port = os.getenv("DB_PORT", "3306")
# # # db_name = os.getenv("DB_NAME", "test")
# # # db_user = os.getenv("DB_USER", "root")
# # # db_password = quote_plus(os.getenv("DB_PASSWORD", ""))

# # # if not google_client_id:
# # #     raise ValueError("GOOGLE_CLIENT_ID is missing in .env")
# # # if not google_client_secret:
# # #     raise ValueError("GOOGLE_CLIENT_SECRET is missing in .env")
# # # if not google_api_key:
# # #     raise ValueError("GOOGLE_API_KEY is missing in .env")

# # # app.config["SQLALCHEMY_DATABASE_URI"] = (
# # #     f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
# # # )
# # # app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# # # db = SQLAlchemy(app)
# # # oauth = OAuth(app)

# # # google = oauth.register(
# # #     name="google",
# # #     client_id=google_client_id,
# # #     client_secret=google_client_secret,
# # #     server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
# # #     client_kwargs={"scope": "openid email profile"},
# # # )

# # # gemini_client = genai.Client(api_key=google_api_key)


# # # class ResumeResult(db.Model):
# # #     __tablename__ = "resume_results"

# # #     id = db.Column(db.Integer, primary_key=True)
# # #     result_id = db.Column(db.String(64), unique=True, nullable=False, index=True)

# # #     user_google_id = db.Column(db.String(255), nullable=True)
# # #     user_name = db.Column(db.String(255), nullable=True)
# # #     user_email = db.Column(db.String(255), nullable=True, index=True)
# # #     user_picture = db.Column(db.Text, nullable=True)

# # #     original_filename = db.Column(db.String(255), nullable=False)
# # #     saved_path = db.Column(db.Text, nullable=False)

# # #     jd_text = db.Column(db.Text, nullable=True)
# # #     resume_data = db.Column(db.JSON, nullable=False)

# # #     theme_color     = db.Column(db.String(20), nullable=False, default="#2980b9")
# # #     saved_template  = db.Column(db.String(32), nullable=True, default="classic")

# # #     created_at = db.Column(db.DateTime, server_default=db.func.now(), nullable=False)
# # #     updated_at = db.Column(db.DateTime, server_default=db.func.now(), onupdate=db.func.now(), nullable=False)


# # # @app.before_request
# # # def create_tables():
# # #     if not getattr(app, "_db_ready", False):
# # #         db.create_all()
# # #         app._db_ready = True


# # # def allowed_file(filename: str) -> bool:
# # #     return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# # # def wait_for_gemini_file(file_obj, timeout=60):
# # #     start = time.time()
# # #     while getattr(file_obj, "state", None) and file_obj.state.name == "PROCESSING":
# # #         if time.time() - start > timeout:
# # #             raise TimeoutError("Gemini file processing timed out.")
# # #         time.sleep(2)
# # #         file_obj = gemini_client.files.get(name=file_obj.name)
# # #     return file_obj


# # # def clean_json_response(text: str) -> str:
# # #     text = text.strip()
# # #     if text.startswith("```json"):
# # #         text = text[7:]
# # #     elif text.startswith("```"):
# # #         text = text[3:]
# # #     if text.endswith("```"):
# # #         text = text[:-3]
# # #     return text.strip()


# # # def ensure_list(value):
# # #     if isinstance(value, list):
# # #         return value
# # #     if value in (None, ""):
# # #         return []
# # #     return [str(value)]


# # # def normalize_resume_data(data):
# # #     if not isinstance(data, dict):
# # #         data = {}

# # #     normalized = {
# # #         "full_name": str(data.get("full_name", "") or ""),
# # #         "headline": str(data.get("headline", "") or ""),
# # #         "career_summary": str(data.get("career_summary", "") or ""),
# # #         "contact": {"phone": "", "email": "", "linkedin": "", "address": ""},
# # #         "work_experience": [],
# # #         "education": [],
# # #         "certifications": [],
# # #         "skills": {"software": [], "languages": [], "other": []},
# # #         "suggestions": [],
# # #     }

# # #     contact = data.get("contact")
# # #     if isinstance(contact, dict):
# # #         normalized["contact"]["phone"]    = str(contact.get("phone", "") or "")
# # #         normalized["contact"]["email"]    = str(contact.get("email", "") or "")
# # #         normalized["contact"]["linkedin"] = str(contact.get("linkedin", "") or "")
# # #         normalized["contact"]["address"]  = str(contact.get("address", "") or "")

# # #     for job in ensure_list(data.get("work_experience")):
# # #         if not isinstance(job, dict):
# # #             continue
# # #         normalized["work_experience"].append({
# # #             "job_title":  str(job.get("job_title", "") or ""),
# # #             "company":    str(job.get("company", "") or ""),
# # #             "location":   str(job.get("location", "") or ""),
# # #             "start_date": str(job.get("start_date", "") or ""),
# # #             "end_date":   str(job.get("end_date", "") or ""),
# # #             "bullets":    [str(x) for x in ensure_list(job.get("bullets")) if str(x).strip()],
# # #         })

# # #     for edu in ensure_list(data.get("education")):
# # #         if not isinstance(edu, dict):
# # #             continue
# # #         normalized["education"].append({
# # #             "institution":    str(edu.get("institution", "") or ""),
# # #             "degree":         str(edu.get("degree", "") or ""),
# # #             "location":       str(edu.get("location", "") or ""),
# # #             "graduation_date": str(edu.get("graduation_date", "") or ""),
# # #             "details":        [str(x) for x in ensure_list(edu.get("details")) if str(x).strip()],
# # #         })

# # #     for cert in ensure_list(data.get("certifications")):
# # #         if not isinstance(cert, dict):
# # #             continue
# # #         normalized["certifications"].append({
# # #             "name":   str(cert.get("name", "") or ""),
# # #             "issuer": str(cert.get("issuer", "") or ""),
# # #             "date":   str(cert.get("date", "") or ""),
# # #         })

# # #     skills = data.get("skills")
# # #     if isinstance(skills, dict):
# # #         normalized["skills"]["software"]  = [str(x) for x in ensure_list(skills.get("software")) if str(x).strip()]
# # #         normalized["skills"]["languages"] = [str(x) for x in ensure_list(skills.get("languages")) if str(x).strip()]
# # #         normalized["skills"]["other"]     = [str(x) for x in ensure_list(skills.get("other")) if str(x).strip()]

# # #     normalized["suggestions"] = [str(x) for x in ensure_list(data.get("suggestions")) if str(x).strip()]

# # #     return normalized


# # # def validate_theme_color(color: str) -> str:
# # #     color = str(color).strip().lower()
# # #     # Handle case where # was stripped by browser URL fragment parsing
# # #     if color and not color.startswith("#"):
# # #         color = "#" + color
# # #     if HEX_COLOR_RE.match(color):
# # #         return color
# # #     return "#2980b9"


# # # def validate_template(template: str) -> str:
# # #     if template in ALLOWED_TEMPLATES:
# # #         return template
# # #     return "classic"


# # # # ─────────────────────────────────────────
# # # #  Routes
# # # # ─────────────────────────────────────────

# # # @app.route("/")
# # # def index():
# # #     return render_template("index.html", user=session.get("user"))


# # # @app.route("/improve")
# # # def improve():
# # #     user = session.get("user")
# # #     if not user:
# # #         return redirect(url_for("login"))
# # #     return render_template("improve.html", user=user)


# # # # Template selection page — shown right after the API returns
# # # @app.route("/select-template/<result_id>")
# # # def select_template(result_id):
# # #     user = session.get("user")
# # #     result = ResumeResult.query.filter_by(result_id=result_id).first()

# # #     if not result:
# # #         return "Result not found", 404

# # #     return render_template(
# # #         "select_template.html",
# # #         user=user,
# # #         result_id=result_id,
# # #     )


# # # # ─────────────────────────────────────────
# # # #  NEW: Step-by-step guided CV editor
# # # #  Flow: select-template → edit → result
# # # # ─────────────────────────────────────────
# # # @app.route("/edit/<result_id>")
# # # def edit_page(result_id):
# # #     user = session.get("user")
# # #     if not user:
# # #         return redirect(url_for("login"))

# # #     result = ResumeResult.query.filter_by(result_id=result_id).first()
# # #     if not result:
# # #         return "Result not found", 404

# # #     # Require a template to be chosen first
# # #     template = request.args.get("template", "").strip()
# # #     if not template:
# # #         return redirect(url_for("select_template", result_id=result_id))

# # #     template = validate_template(template)

# # #     color = request.args.get("color", "").strip()
# # #     theme = validate_theme_color(color) if color else (result.theme_color or "#ba7517")

# # #     # Persist chosen theme color
# # #     if color:
# # #         validated_color = validate_theme_color(color)
# # #         if validated_color != result.theme_color:
# # #             result.theme_color = validated_color
# # #             db.session.commit()

# # #     safe_resume_data = normalize_resume_data(result.resume_data)

# # #     return render_template(
# # #         "edit.html",
# # #         user=user,
# # #         result=result,
# # #         cv=safe_resume_data,
# # #         theme=theme,
# # #         template=template,
# # #         result_id=result_id,
# # #     )


# # # # Result page — renders the chosen template
# # # @app.route("/result/<result_id>")
# # # def result_page(result_id):
# # #     user = session.get("user")
# # #     result = ResumeResult.query.filter_by(result_id=result_id).first()

# # #     if not result:
# # #         return "Result not found", 404

# # #     # If no template chosen yet, redirect to template picker
# # #     template = request.args.get("template", "").strip()
# # #     if not template:
# # #         return redirect(url_for("select_template", result_id=result_id))

# # #     template = validate_template(template)

# # #     # Color: prefer query param, fall back to saved value
# # #     color = request.args.get("color", "").strip()
# # #     theme = validate_theme_color(color) if color else (result.theme_color or "#ba7517")

# # #     # Persist the chosen theme color
# # #     if color and validate_theme_color(color) != result.theme_color:
# # #         result.theme_color = validate_theme_color(color)
# # #         db.session.commit()

# # #     safe_resume_data = normalize_resume_data(result.resume_data)

# # #     return render_template(
# # #         "result.html",
# # #         user=user,
# # #         result=result,
# # #         cv=safe_resume_data,
# # #         theme=theme,
# # #         template=template,
# # #         result_id=result_id,
# # #     )


# # # @app.route("/api/improve-cv", methods=["POST"])
# # # def improve_cv_api():
# # #     user = session.get("user")
# # #     if not user:
# # #         return jsonify({"ok": False, "error": "Please log in first."}), 401

# # #     if "cv_file" not in request.files:
# # #         return jsonify({"ok": False, "error": "CV file is required."}), 400

# # #     cv_file = request.files["cv_file"]

# # #     if cv_file.filename == "":
# # #         return jsonify({"ok": False, "error": "No file selected."}), 400

# # #     if not allowed_file(cv_file.filename):
# # #         return jsonify({"ok": False, "error": "Unsupported CV file type."}), 400

# # #     jd_text = request.form.get("jd_text", "").strip()

# # #     cv_file.seek(0, os.SEEK_END)
# # #     size = cv_file.tell()
# # #     cv_file.seek(0)

# # #     if size > MAX_FILE_SIZE:
# # #         return jsonify({"ok": False, "error": "CV file must be under 10 MB."}), 400

# # #     safe_name = secure_filename(cv_file.filename)
# # #     unique_name = f"{uuid.uuid4().hex}_{safe_name}"
# # #     saved_path = os.path.join(UPLOAD_FOLDER, unique_name)
# # #     cv_file.save(saved_path)

# # #     try:
# # #         uploaded_cv = gemini_client.files.upload(file=saved_path)
# # #         uploaded_cv = wait_for_gemini_file(uploaded_cv)

# # #         if not getattr(uploaded_cv, "state", None) or uploaded_cv.state.name != "ACTIVE":
# # #             state_name = getattr(getattr(uploaded_cv, "state", None), "name", "UNKNOWN")
# # #             return jsonify({"ok": False, "error": f"Gemini file processing failed: {state_name}"}), 500

# # #         prompt = f"""
# # # You are an expert ATS resume writer.

# # # Read the uploaded CV and the user details. Rewrite the resume into a clean, ATS-friendly format.

# # # IMPORTANT RULES:
# # # - Return ONLY valid JSON.
# # # - Do not wrap the JSON in markdown or triple backticks.
# # # - Do not include explanations outside JSON.
# # # - Keep wording concise and achievement-focused.
# # # - Use strong action verbs.
# # # - Tailor the content to the provided job description if available.
# # # - Preserve truthfulness. Do not invent experience.

# # # User details:
# # # - Name: {user.get("name", "")}
# # # - Email: {user.get("email", "")}

# # # Job Description:
# # # {jd_text if jd_text else "No job description provided."}

# # # Return JSON in exactly this structure:
# # # {{
# # #   "full_name": "",
# # #   "headline": "",
# # #   "contact": {{
# # #     "phone": "",
# # #     "email": "",
# # #     "linkedin": "",
# # #     "address": ""
# # #   }},
# # #   "career_summary": "",
# # #   "work_experience": [
# # #     {{
# # #       "job_title": "",
# # #       "company": "",
# # #       "location": "",
# # #       "start_date": "",
# # #       "end_date": "",
# # #       "bullets": [""]
# # #     }}
# # #   ],
# # #   "education": [
# # #     {{
# # #       "institution": "",
# # #       "degree": "",
# # #       "location": "",
# # #       "graduation_date": "",
# # #       "details": [""]
# # #     }}
# # #   ],
# # #   "certifications": [
# # #     {{
# # #       "name": "",
# # #       "issuer": "",
# # #       "date": ""
# # #     }}
# # #   ],
# # #   "skills": {{
# # #     "software": [""],
# # #     "languages": [""],
# # #     "other": [""]
# # #   }},
# # #   "suggestions": [""]
# # # }}
# # # """

# # #         response = gemini_client.models.generate_content(
# # #             model=MODEL_ID,
# # #             contents=[uploaded_cv, prompt],
# # #         )

# # #         raw_text = response.text or ""
# # #         cleaned_text = clean_json_response(raw_text)
# # #         resume_data = json.loads(cleaned_text)
# # #         resume_data = normalize_resume_data(resume_data)

# # #         result_id = uuid.uuid4().hex

# # #         new_result = ResumeResult(
# # #             result_id=result_id,
# # #             user_google_id=user.get("id"),
# # #             user_name=user.get("name"),
# # #             user_email=user.get("email"),
# # #             user_picture=user.get("picture"),
# # #             original_filename=cv_file.filename,
# # #             saved_path=saved_path,
# # #             jd_text=jd_text,
# # #             resume_data=resume_data,
# # #             theme_color="#2980b9",
# # #         )

# # #         db.session.add(new_result)
# # #         db.session.commit()

# # #         # Clean up local upload after Gemini has processed it
# # #         try:
# # #             os.remove(saved_path)
# # #         except OSError:
# # #             pass

# # #         # Redirect to template picker
# # #         return jsonify({
# # #             "ok": True,
# # #             "message": "CV processed successfully.",
# # #             "result_id": result_id,
# # #             "redirect_url": url_for("select_template", result_id=result_id),
# # #         })

# # #     except json.JSONDecodeError:
# # #         return jsonify({"ok": False, "error": "Gemini returned invalid JSON. Please try again."}), 500
# # #     except Exception as e:
# # #         return jsonify({"ok": False, "error": str(e)}), 500


# # # @app.route("/api/result/<result_id>", methods=["GET"])
# # # def get_result_data(result_id):
# # #     result = ResumeResult.query.filter_by(result_id=result_id).first()
# # #     if not result:
# # #         return jsonify({"ok": False, "error": "Result not found"}), 404

# # #     return jsonify({
# # #         "ok": True,
# # #         "result_id": result_id,
# # #         "resume_data": normalize_resume_data(result.resume_data),
# # #         "theme_color": result.theme_color or "#2980b9",
# # #     })


# # # @app.route("/api/result/<result_id>/update", methods=["POST"])
# # # def update_result_data(result_id):
# # #     user = session.get("user")
# # #     if not user:
# # #         return jsonify({"ok": False, "error": "Please log in first."}), 401

# # #     result = ResumeResult.query.filter_by(result_id=result_id).first()
# # #     if not result:
# # #         return jsonify({"ok": False, "error": "Result not found"}), 404

# # #     if result.user_google_id and result.user_google_id != user.get("id"):
# # #         return jsonify({"ok": False, "error": "Forbidden"}), 403

# # #     data = request.get_json(silent=True) or {}

# # #     if data.get("resume_data") is not None:
# # #         result.resume_data = normalize_resume_data(data["resume_data"])

# # #     if data.get("theme_color"):
# # #         result.theme_color = validate_theme_color(data["theme_color"])

# # #     if data.get("saved_template"):
# # #         result.saved_template = validate_template(data["saved_template"])

# # #     db.session.commit()
# # #     return jsonify({"ok": True, "message": "Resume updated successfully."})


# # # @app.route("/login")
# # # def login():
# # #     redirect_uri = url_for("callback", _external=True)
# # #     return google.authorize_redirect(redirect_uri)


# # # @app.route("/callback")
# # # def callback():
# # #     token = google.authorize_access_token()
# # #     user_info = token.get("userinfo")
# # #     if not user_info:
# # #         user_info = google.parse_id_token(token)

# # #     session["user"] = {
# # #         "id":      user_info.get("sub"),
# # #         "name":    user_info.get("name"),
# # #         "email":   user_info.get("email"),
# # #         "picture": user_info.get("picture"),
# # #     }
# # #     # If user has existing CVs, go to documents page; else improve
# # #     existing = ResumeResult.query.filter_by(user_google_id=user_info.get("sub")).first()
# # #     if existing:
# # #         return redirect(url_for("documents"))
# # #     return redirect(url_for("improve"))


# # # @app.route("/logout")
# # # def logout():
# # #     session.clear()
# # #     return redirect(url_for("index"))


# # # # ─────────────────────────────────────────
# # # #  Documents page — user's saved CVs
# # # # ─────────────────────────────────────────
# # # @app.route("/documents")
# # # def documents():
# # #     user = session.get("user")
# # #     if not user:
# # #         return redirect(url_for("login"))

# # #     resumes = ResumeResult.query.filter_by(
# # #         user_google_id=user.get("id")
# # #     ).order_by(ResumeResult.updated_at.desc()).all()

# # #     return render_template("documents.html", user=user, resumes=resumes)


# # # # ─────────────────────────────────────────
# # # #  Delete a CV
# # # # ─────────────────────────────────────────
# # # @app.route("/api/result/<result_id>/delete", methods=["DELETE"])
# # # def delete_result(result_id):
# # #     user = session.get("user")
# # #     if not user:
# # #         return jsonify({"ok": False, "error": "Please log in first."}), 401

# # #     result = ResumeResult.query.filter_by(result_id=result_id).first()
# # #     if not result:
# # #         return jsonify({"ok": False, "error": "Not found"}), 404

# # #     if result.user_google_id and result.user_google_id != user.get("id"):
# # #         return jsonify({"ok": False, "error": "Forbidden"}), 403

# # #     db.session.delete(result)
# # #     db.session.commit()
# # #     return jsonify({"ok": True})


# # # if __name__ == "__main__":
# # #     app.run(debug=True)
