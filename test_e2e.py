"""End-to-end test of the whole InfraPulse flow. Run:  python test_e2e.py"""

import os, shutil, sys, tempfile

WORK = tempfile.mkdtemp(prefix="infrapulse_test_")
os.environ["INFRAPULSE_DB"] = os.path.join(WORK, "test.db")
os.environ["INFRAPULSE_UPLOADS"] = os.path.join(WORK, "uploads")
os.environ["INFRAPULSE_MODEL"] = os.path.join(WORK, "model.pt")
os.environ["INFRAPULSE_SECRET"] = "test-secret"
os.makedirs(os.environ["INFRAPULSE_UPLOADS"], exist_ok=True)

import numpy as np, torch, torch.nn as nn
from PIL import Image
from torchvision import models

CLASSES = ["cracked_tiles", "paint_peeling", "spalling", "stagnant_water"]

# A throwaway checkpoint with the same shape the notebook produces, so we exercise the
# real loading / Grad-CAM / severity path without needing the trained weights.
net = models.resnet18(weights=None)
net.fc = nn.Linear(net.fc.in_features, len(CLASSES))
torch.save({"arch": "resnet18", "state_dict": net.state_dict(), "classes": CLASSES,
            "img_size": 224, "norm_mean": [0.485, 0.456, 0.406],
            "norm_std": [0.229, 0.224, 0.225], "test_accuracy": 0.0, "macro_f1": 0.0},
           os.environ["INFRAPULSE_MODEL"])

from fastapi.testclient import TestClient
from app import db
from app.ml import priority
from app.main import app

PASS, FAIL = [], []
def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {detail}" if detail and not cond else ""))

def make_photo(path, seed):
    rng = np.random.default_rng(seed)
    a = rng.integers(90, 190, (420, 420, 3), dtype=np.uint8)
    a[120:300, 90:330] = rng.integers(0, 70, (180, 240, 3), dtype=np.uint8)  # a "defect" patch
    Image.fromarray(a).save(path)
    return path

print("\n--- priority formula ---")
p_tile = priority.compute_priority("cracked_tiles", 0.6, 0.4)
p_paint = priority.compute_priority("paint_peeling", 0.6, 0.4)
check("cracked tiles outrank paint peeling at identical severity and extent",
      p_tile["priority_score"] > p_paint["priority_score"],
      f"{p_tile['priority_score']} vs {p_paint['priority_score']}")
check("formula is deterministic",
      priority.compute_priority("spalling", .5, .5) == priority.compute_priority("spalling", .5, .5))
check("higher severity scores higher",
      priority.compute_priority("spalling", .9, .5)["priority_score"] >
      priority.compute_priority("spalling", .2, .5)["priority_score"])
check("higher extent scores higher",
      priority.compute_priority("spalling", .5, .9)["priority_score"] >
      priority.compute_priority("spalling", .5, .1)["priority_score"])
check("score matches the documented arithmetic",
      abs(priority.compute_priority("cracked_tiles", 0.6, 0.4)["priority_score"]
          - 100 * 1.0 * (0.5 * 0.6 + 0.5 * 0.4)) < 0.01)

print("\n--- web flow ---")
client = TestClient(app)
with client:
    r = client.get("/")
    check("home page loads", r.status_code == 200)
    check("health reports the model loaded", client.get("/health").json()["model"]["loaded"])

    r = client.post("/register", data={"name": "Asha R", "email": "asha@test.in",
                                       "password": "secret123"}, follow_redirects=True)
    check("resident can register and lands on dashboard", "My complaints" in r.text)

    r = client.post("/register", data={"name": "X", "email": "asha@test.in",
                                       "password": "secret123"})
    check("duplicate email is rejected", "already registered" in r.text)

    photo = make_photo(os.path.join(WORK, "p1.jpg"), 1)
    with open(photo, "rb") as f:
        r = client.post("/complaints/new",
                        data={"reporter_name": "Asha R", "address": "Hall 6, B-wing",
                              "description": "Wall damage near the stairs"},
                        files={"photo": ("p1.jpg", f, "image/jpeg")}, follow_redirects=True)
    check("complaint submits and shows the detection", "What the system detected" in r.text)
    check("a defect name is displayed", any(n in r.text for n in
          ["Cracked Tiles", "Paint Peeling", "Spalling", "Stagnant Water"]))
    check("a category is displayed", any(c in r.text for c in
          ["Structural", "Functional", "Performance"]))

    row = db.get_complaint(1)
    check("analysis was stored", row is not None and row["defect"] in CLASSES)
    check("category routing matches the defect",
          row["category"] == {"spalling": "Structural", "stagnant_water": "Functional",
                              "cracked_tiles": "Performance",
                              "paint_peeling": "Performance"}[row["defect"]])
    check("severity is in range", 0.0 <= row["severity"] <= 1.0)
    check("extent is in range", 0.0 <= row["extent"] <= 1.0)
    check("priority score is positive", row["priority_score"] > 0)
    check("status starts at Submitted", row["status"] == "Submitted")

    r = client.post("/complaints/new",
                    data={"reporter_name": "Asha R", "address": "x", "description": "y"},
                    files={"photo": ("notes.txt", b"hello", "text/plain")})
    check("non-image upload is refused", "upload a photograph" in r.text)

print("\n--- queue ordering, isolation, status pipeline ---")
# Craft complaints with known scores so ordering is unambiguous.
uid = db.get_user_by_email("asha@test.in")["id"]
def add(defect, sev, ext, addr):
    a = {"defect": defect,
         "defect_name": defect.replace("_", " ").title(),
         "category": {"spalling": "Structural", "stagnant_water": "Functional",
                      "cracked_tiles": "Performance", "paint_peeling": "Performance"}[defect],
         "confidence": 0.9, "severity": sev, "extent": ext}
    return db.create_complaint(uid, "Asha R", addr, "d", "p1.jpg", a,
                               priority.compute_priority(defect, sev, ext))

low  = add("cracked_tiles", 0.10, 0.10, "low one")
high = add("cracked_tiles", 0.95, 0.90, "high one")
mid  = add("paint_peeling", 0.80, 0.80, "mid one")
struct = add("spalling", 0.70, 0.70, "structural one")

perf = db.queue_for_category("Performance")
ids = [c["id"] for c in perf]
check("highest priority sits at the top of the queue", ids[0] == high)
check("lowest priority sits at the bottom", ids[-1] == low)
check("queue is sorted by priority descending",
      all(perf[i]["priority_score"] >= perf[i+1]["priority_score"] for i in range(len(perf)-1)))
check("a newly added complaint is placed automatically, not appended blindly",
      ids.index(mid) < ids.index(low))
check("Structural queue holds only structural complaints",
      all(c["category"] == "Structural" for c in db.queue_for_category("Structural")))
check("a Performance complaint never appears in the Structural queue",
      high not in [c["id"] for c in db.queue_for_category("Structural")])

ok, _ = db.update_status(high, "In Progress", "T")
check("status cannot skip a step", not ok)
ok, _ = db.update_status(high, "Assigned", "T")
check("Submitted -> Assigned is allowed", ok)
ok, _ = db.update_status(high, "Submitted", "T")
check("status cannot move backwards", not ok)
db.update_status(high, "In Progress", "T")
ok, _ = db.update_status(high, "Resolved", "T")
check("full pipeline reaches Resolved", ok)

check("resolved complaint leaves the live queue",
      high not in [c["id"] for c in db.queue_for_category("Performance")])
check("resolved complaint stays in the user's history",
      high in [c["id"] for c in db.complaints_for_user(uid)])
check("resolved complaint has no queue position", db.queue_position(high) is None)
check("remaining queue re-ranks after a resolution",
      db.queue_for_category("Performance")[0]["id"] == mid)
check("status history was recorded",
      [h["status"] for h in db.history_for(high)] ==
      ["Submitted", "Assigned", "In Progress", "Resolved"])

print("\n--- staff portal ---")
db.init_db()
from app import security
db.create_user("perf@test.in", "Perf Team", security.hash_password("staff1234"),
               role="staff", category="Performance")
sc = TestClient(app)
with sc:
    r = sc.post("/login", data={"email": "perf@test.in", "password": "staff1234"},
                follow_redirects=True)
    check("staff login lands on their own queue", "Performance queue" in r.text)
    check("staff see their category's complaints", "mid one" in r.text)
    check("staff do not see another category's complaints", "structural one" not in r.text)
    check("staff API is scoped to their category",
          sc.get("/api/queue/Structural").status_code == 403)
    r = sc.get(f"/complaints/{struct}")
    check("staff cannot open a complaint outside their category",
          "Different category" in r.text)
    r = sc.post(f"/staff/complaints/{mid}/status", data={"new_status": "Assigned"},
                follow_redirects=True)
    check("staff can advance a complaint", db.get_complaint(mid)["status"] == "Assigned")

shutil.rmtree(WORK, ignore_errors=True)
print(f"\n{'='*58}\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", *FAIL, sep="\n  - ")
sys.exit(1 if FAIL else 0)
