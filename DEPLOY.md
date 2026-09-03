# Deploying InfraPulse

The problem statement requires a **working URL that stays reachable throughout the
evaluation period**, so host choice is a correctness decision.

> **Note on Hugging Face Spaces:** Docker and Gradio Spaces now require a paid PRO plan.
> Only Static Spaces remain free, and those cannot run a Python backend. Use Render.

---

## Render (primary)

Free Docker web services, no card required.

**1. Sign in**

render.com → **Get Started** → sign in **with GitHub** (simplest, and it can then see your
repository directly).

**2. Create the service**

- Dashboard → **New +** → **Web Service**
- Connect the `infrapulse` repository. If it isn't listed, click **Configure account** and
  grant Render access to it — a private repo works fine.
- Render reads `render.yaml` from the repo and fills most of this in. Confirm:

| Field | Value |
|---|---|
| Name | `infrapulse` |
| Language / Runtime | **Docker** |
| Branch | `main` |
| Instance type | **Free** |
| Health check path | `/health` |

**3. Environment variables**

Under **Environment**, add:

| Key | Value |
|---|---|
| `INFRAPULSE_SECRET` | any long random string |
| `STAFF_PASSWORD` | the staff password you want |

`INFRAPULSE_SECRET` matters — without it the app generates a fresh signing key on every
restart, which silently logs everyone out.

**4. Deploy**

Click **Create Web Service**. The first build takes 5–10 minutes; PyTorch is large. Watch
the log stream. Success looks like:

```
[InfraPulse] model loaded: {'loaded': True, 'classes': [...]}
INFO:  Uvicorn running on http://0.0.0.0:10000
==> Your service is live 🎉
```

**5. Verify before calling it done**

- `https://infrapulse-XXXX.onrender.com/health` → must report `"loaded": true`
- Open the site in a **private/incognito window** — you are then testing as an evaluator
  with no session. Register, submit a photo, confirm a defect and category come back.
- Log in as staff, confirm the complaint is in the right queue.

### Two things to know about the free tier

**It sleeps after 15 minutes idle.** The next visitor waits roughly 50 seconds for a cold
start. Before the evaluation window, open the URL yourself to wake it, and keep a tab
loading it every few minutes.

**512 MB RAM is tight for PyTorch.** The app sets `torch.set_num_threads(1)` and
`cv2.setNumThreads(1)` to keep the footprint down. If the logs show `Out of memory` or the
service restarts under load, use the fallback below.

---

## Fallback: Cloudflare Tunnel from your own machine

If Render runs out of memory, this exposes the app running on your laptop at a public URL.
No account, no card, works in two minutes.

**1. Download** `cloudflared` for Windows:
https://github.com/cloudflare/cloudflared/releases/latest — take
`cloudflared-windows-amd64.exe`, rename it `cloudflared.exe`.

**2. Run the app** in one PowerShell window:

```powershell
cd "C:\Users\ADMIN\OneDrive\Desktop\Descon Mid_prep\infrapulse"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**3. Open the tunnel** in a second window:

```powershell
.\cloudflared.exe tunnel --url http://localhost:8000
```

It prints a public URL like `https://random-words-here.trycloudflare.com`. That link works
for anyone, anywhere.

**The catch:** your laptop must stay awake, online, and running both windows for the whole
evaluation period, and the URL changes each time you restart the tunnel. Acceptable as a
backup or for a live demo; not ideal as the primary submission link.

If you use this, disable sleep: **Settings → System → Power → Screen and sleep → Never**.

---

## Before the deadline

- [ ] `/health` returns `"loaded": true`
- [ ] Full flow works in a private window on the live URL
- [ ] Staff accounts exist, each seeing only its own category
- [ ] URL is public — no login wall
- [ ] `INFRAPULSE_SECRET` is set
- [ ] Someone else opens the link on their phone and it works

That last one is the cheapest way to catch a URL that only works because your own browser
happens to hold a session.
