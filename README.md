# InfraPulse
🔗<https://infrapulse-8pd0.onrender.com>

**Photo-Based Defect Detection & Priority Maintenance Web System**
Team: **Shauryas** — Takneek'26

A resident photographs a building defect and submits it. InfraPulse identifies the defect,
classifies it into a maintenance category, scores how urgent it is from what is visible in
the photograph, and places it in the correct staff queue at the correct position — with no
manual categorisation or re-sorting at any point.

---

## What it does

| Stage | Behaviour |
|---|---|
| Register / login | Separate portals for residents and staff |
| Submit | Name, address/location, description, photograph |
| Detect | Our fine-tuned CNN identifies the visible defect |
| Classify | Defect is mapped to Structural / Functional / Performance |
| Prioritise | Score computed from visible severity and visible extent |
| Route | Complaint enters that category's queue, in priority order |
| Track | Submitted → Assigned → In Progress → Resolved |
| Resolve | Leaves the live queue, stays in the resident's history |

**Defect → category routing**

| Defect | Category |
|---|---|
| Spalling | Structural |
| Stagnant Water | Functional |
| Cracked Tiles | Performance |
| Paint Peeling | Performance |

---

## Setup

Requires Python 3.10 or newer.

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. put the trained model in place
#    (produced by InfraPulse_Train.ipynb — see "Training the model" below)
#    model/infrapulse_model.pt

# 3. create the three staff accounts, one per category
python seed_staff.py

# 4. run
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>.

### Accounts

Residents register themselves at `/register`.

Staff accounts are created by `seed_staff.py`:

| Email | Category |
|---|---|
| structural@infrapulse.local | Structural |
| functional@infrapulse.local | Functional |
| performance@infrapulse.local | Performance |

Default password `staff1234`, override with the `STAFF_PASSWORD` environment variable.
Each staff account can see and act on **only** its own category's queue.

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `INFRAPULSE_MODEL` | `model/infrapulse_model.pt` | Trained checkpoint |
| `INFRAPULSE_DB` | `infrapulse.db` | SQLite database file |
| `INFRAPULSE_UPLOADS` | `uploads` | Submitted photographs |
| `INFRAPULSE_SECRET` | random per start | Session signing key — **set this in production** |
| `STAFF_PASSWORD` | `staff1234` | Password for seeded staff accounts |

---

## Training the model

`InfraPulse_Train.ipynb` runs in Google Colab and does the whole ML pipeline: assembles the
dataset from public sources, fine-tunes an **ImageNet**-pretrained ResNet18 on the four
defect classes, evaluates it (confusion matrix, precision, recall, F1), and exports
`infrapulse_model.pt`, `metrics.json` and `confusion_matrix.png`.

Drop `infrapulse_model.pt` into `model/` and restart the server.

---

## How priority is calculated

```
priority_score = 100 × base_weight × (0.5 × severity + 0.5 × extent)
```

- **severity** (0–1) — measured inside the defect region only: local contrast, edge
  density, and how far the region's intensity deviates from the surrounding surface.
- **extent** (0–1) — the fraction of the photograph the defect occupies, obtained by
  running **Grad-CAM** on our own trained network and thresholding the resulting heatmap.
  No second model and no extra annotation is required.
- **base_weight** — `cracked_tiles` 1.00, `paint_peeling` 0.75, `spalling` 1.00,
  `stagnant_water` 1.00. Because each category owns a separate queue, a base weight only
  ever affects ordering *within* a category; Performance is the only category with two
  defect types, and the problem statement fixes cracked tiles above paint peeling.

Equal scores are broken by submission time, oldest first.

Every value is computed from the submitted photograph at request time. The formula is
deterministic — the same photograph always produces the same score — and the queue sorts
by exactly this number. `app/ml/priority.py` is the single source of truth.

---

## Compliance with the problem statement

- **No external AI/ML inference service or API is used for detection or classification.**
  Inference runs inside this application, on CPU, using weights we trained ourselves.
- The only pretrained starting point is **ResNet18 pretrained on ImageNet**, a
  general-purpose dataset, explicitly permitted as a backbone. No checkpoint pretrained on
  defect, crack, stagnant-water or building-damage data is used anywhere.
- Nothing is hardcoded, memorised, or pre-associated with any expected evaluation input.
  Every complaint is analysed live from its uploaded photograph.
- Non-AI supporting libraries (FastAPI, SQLite, Jinja2, OpenCV, Pillow) are used for
  serving, storage, templating and basic image measurement only.

---

## Project layout

```
InfraPulse_Train.ipynb   dataset assembly, training, evaluation (run in Google Colab)
DOCUMENTATION.md         the full report: approach, detection logic, priority method,
                         evaluation results, limitations, suggested improvements
DEPLOY.md                hosting instructions
app/
  main.py              FastAPI routes — both portals and the JSON API
  db.py                SQLite schema, queue ordering, status pipeline
  security.py          PBKDF2 password hashing (standard library only)
  ml/
    analyzer.py        model loading, classification, Grad-CAM, severity
    priority.py        the documented priority formula
  templates/           server-rendered pages
  static/style.css
model/
  infrapulse_model.pt  the trained checkpoint
  metrics.json         evaluation scores
  confusion_matrix.png test-set confusion matrix
uploads/               submitted photographs
seed_staff.py          creates the three staff accounts
test_e2e.py            end-to-end test suite
```

---

## Tests

```bash
python test_e2e.py
```

Covers the priority formula, registration and submission, defect→category routing,
queue ordering, automatic placement of new complaints, category isolation between staff,
the forward-only status pipeline, and removal of resolved complaints from the live queue
while retaining them in the resident's history.

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Confirms the model is loaded and serving |
| `GET /api/queue/{category}` | Live queue for a category, in priority order |
| `GET /api/my-complaints` | The signed-in resident's complaints and queue positions |

The portals poll these so queues update on their own as new complaints arrive.

---

## Known limitations

- Training images come from several public datasets with different capture conditions, so
  the model can partly key on source-specific appearance rather than the defect alone.
  Mitigated with region cropping, heavy augmentation and a held-out test split.
- Severity is a visual proxy. It cannot judge structural depth from a single photograph.
- Grad-CAM gives a coarse localisation, so extent is approximate for small thin defects.
- Uploaded photographs are stored on local disk; on an ephemeral host they do not survive
  a restart.
