"""
InfraPulse — Photo-Based Defect Detection & Priority Maintenance Web System.

One FastAPI application serves both portals and the API. Pages are server-rendered with
Jinja2 so there is a single thing to deploy and a single URL for evaluators.
"""

from __future__ import annotations

import os
import secrets
import uuid

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from app import db, security
from app.ml import analyzer, priority

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.environ.get("INFRAPULSE_UPLOADS", "uploads")
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="InfraPulse")
app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get("INFRAPULSE_SECRET", secrets.token_hex(32)),
    same_site="lax",
)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
app.mount("/photos", StaticFiles(directory=UPLOAD_DIR), name="photos")

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
templates.env.globals["CATEGORIES"] = analyzer.CATEGORIES


@app.on_event("startup")
def _startup() -> None:
    db.init_db()
    try:
        analyzer.load_model()
        print("[InfraPulse] model loaded:", analyzer.model_info())
    except analyzer.ModelNotLoaded as e:
        print("[InfraPulse] WARNING —", e)
        print("[InfraPulse] The site will run, but complaint submission will refuse "
              "until the model file is in place.")


# ----------------------------------------------------------------- helpers
def current_user(request: Request):
    uid = request.session.get("uid")
    return db.get_user(uid) if uid else None


def render(request: Request, template: str, **ctx):
    return templates.TemplateResponse(
        request, template, {"user": current_user(request), **ctx}
    )


def need_login(request: Request, role: str | None = None):
    """Returns a redirect if the visitor may not see this page, else None."""
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    if role and user["role"] != role:
        dest = "/staff" if user["role"] == "staff" else "/dashboard"
        return RedirectResponse(dest, status_code=303)
    return None


def row_to_dict(row) -> dict:
    return {k: row[k] for k in row.keys()}


# ----------------------------------------------------------------- auth
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    user = current_user(request)
    if user:
        return RedirectResponse("/staff" if user["role"] == "staff" else "/dashboard",
                                status_code=303)
    return render(request, "home.html", stats=db.stats())


@app.get("/register", response_class=HTMLResponse)
def register_form(request: Request):
    return render(request, "register.html")


@app.post("/register")
def register(request: Request, name: str = Form(...), email: str = Form(...),
             password: str = Form(...)):
    if len(password) < 6:
        return render(request, "register.html", error="Password must be at least 6 characters.",
                      name=name, email=email)
    if db.get_user_by_email(email):
        return render(request, "register.html", error="That email is already registered.",
                      name=name, email=email)

    uid = db.create_user(email, name, security.hash_password(password), role="user")
    request.session["uid"] = uid
    return RedirectResponse("/dashboard", status_code=303)


@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    return render(request, "login.html")


@app.post("/login")
def login(request: Request, email: str = Form(...), password: str = Form(...)):
    row = db.get_user_by_email(email)
    if row is None or not security.verify_password(password, row["password_hash"]):
        return render(request, "login.html", error="Wrong email or password.", email=email)

    request.session["uid"] = row["id"]
    return RedirectResponse("/staff" if row["role"] == "staff" else "/dashboard",
                            status_code=303)


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/", status_code=303)


# ----------------------------------------------------------------- user portal
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    if (r := need_login(request, "user")):
        return r
    user = current_user(request)
    rows = db.complaints_for_user(user["id"])
    items = []
    for row in rows:
        d = row_to_dict(row)
        d["position"] = db.queue_position(row["id"])
        items.append(d)
    return render(request, "dashboard.html", complaints=items)


@app.get("/complaints/new", response_class=HTMLResponse)
def new_complaint_form(request: Request):
    if (r := need_login(request, "user")):
        return r
    return render(request, "submit.html", model_ready=analyzer.model_info()["loaded"])


@app.post("/complaints/new")
async def new_complaint(request: Request, reporter_name: str = Form(...),
                        address: str = Form(...), description: str = Form(...),
                        photo: UploadFile = File(...)):
    if (r := need_login(request, "user")):
        return r
    user = current_user(request)

    def fail(msg):
        return render(request, "submit.html", error=msg,
                      model_ready=analyzer.model_info()["loaded"],
                      reporter_name=reporter_name, address=address, description=description)

    ext = os.path.splitext(photo.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        return fail("Please upload a photograph (.jpg, .png, .webp or .bmp).")

    data = await photo.read()
    if not data:
        return fail("That file was empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        return fail("Photograph is larger than 12 MB. Please upload a smaller image.")

    fname = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(UPLOAD_DIR, fname)
    with open(path, "wb") as f:
        f.write(data)

    # ---- the automatic pipeline: detect -> classify -> score -> route ----------------
    try:
        analysis = analyzer.analyze(path)
    except analyzer.ModelNotLoaded as e:
        os.remove(path)
        return fail(str(e))
    except Exception as e:
        os.remove(path)
        return fail(f"That image could not be analysed ({type(e).__name__}). "
                    f"Please try a different photograph.")

    prio = priority.compute_priority(analysis["defect"], analysis["severity"],
                                     analysis["extent"])
    cid = db.create_complaint(user["id"], reporter_name, address, description, fname,
                              analysis, prio)
    return RedirectResponse(f"/complaints/{cid}", status_code=303)


@app.get("/complaints/{complaint_id}", response_class=HTMLResponse)
def complaint_detail(request: Request, complaint_id: int):
    if (r := need_login(request)):
        return r
    user = current_user(request)
    row = db.get_complaint(complaint_id)
    if row is None:
        return render(request, "message.html", title="Not found",
                      message="No complaint with that number.")

    # a user sees only their own; staff see only their own category
    if user["role"] == "user" and row["user_id"] != user["id"]:
        return render(request, "message.html", title="Not your complaint",
                      message="You can only view complaints you filed.")
    if user["role"] == "staff" and row["category"] != user["category"]:
        return render(request, "message.html", title="Different category",
                      message=f"That complaint belongs to the {row['category']} queue.")

    return render(request, "detail.html", c=row_to_dict(row),
                  position=db.queue_position(complaint_id),
                  history=[row_to_dict(h) for h in db.history_for(complaint_id)],
                  next_status=db.next_status(row["status"]))


# ----------------------------------------------------------------- staff portal
@app.get("/staff", response_class=HTMLResponse)
def staff_queue(request: Request):
    if (r := need_login(request, "staff")):
        return r
    user = current_user(request)
    cat = user["category"]
    return render(request, "staff.html", category=cat,
                  queue=[row_to_dict(x) for x in db.queue_for_category(cat)],
                  resolved=[row_to_dict(x) for x in db.resolved_for_category(cat)],
                  statuses=db.STATUSES)


@app.post("/staff/complaints/{complaint_id}/status")
def staff_update_status(request: Request, complaint_id: int, new_status: str = Form(...)):
    if (r := need_login(request, "staff")):
        return r
    user = current_user(request)
    row = db.get_complaint(complaint_id)
    if row is None or row["category"] != user["category"]:
        return RedirectResponse("/staff?error=Not+in+your+category", status_code=303)

    ok, msg = db.update_status(complaint_id, new_status, user["name"])
    return RedirectResponse(f"/staff?{'msg' if ok else 'error'}={msg.replace(' ', '+')}",
                            status_code=303)


# ----------------------------------------------------------------- JSON API
# The portals poll these so queues refresh on their own as new complaints arrive.
@app.get("/api/queue/{category}")
def api_queue(request: Request, category: str):
    user = current_user(request)
    if user is None or (user["role"] == "staff" and user["category"] != category):
        return JSONResponse({"error": "forbidden"}, status_code=403)

    rows = db.queue_for_category(category)
    return {
        "category": category,
        "count": len(rows),
        "queue": [
            {"position": i, "id": r["id"], "defect_name": r["defect_name"],
             "category": r["category"], "priority_score": r["priority_score"],
             "priority_band": r["priority_band"], "status": r["status"],
             "address": r["address"], "created_at": r["created_at"]}
            for i, r in enumerate(rows, start=1)
        ],
    }


@app.get("/api/my-complaints")
def api_my_complaints(request: Request):
    user = current_user(request)
    if user is None or user["role"] != "user":
        return JSONResponse({"error": "forbidden"}, status_code=403)
    return {
        "complaints": [
            {"id": r["id"], "defect_name": r["defect_name"], "category": r["category"],
             "priority_score": r["priority_score"], "priority_band": r["priority_band"],
             "status": r["status"], "position": db.queue_position(r["id"]),
             "created_at": r["created_at"]}
            for r in db.complaints_for_user(user["id"])
        ]
    }


@app.get("/health")
def health():
    """Proves the real model is loaded and serving — handy during evaluation."""
    return {"status": "ok", "model": analyzer.model_info(), "stats": db.stats()}
