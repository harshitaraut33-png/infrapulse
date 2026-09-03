# Deploying InfraPulse

The problem statement requires a **working URL that stays reachable throughout the
evaluation period**. That makes host choice a correctness decision, not a preference.

---

## Why Hugging Face Spaces

Measured footprint of this app: PyTorch's import alone dominates memory, and a running
request peaks well above what a 512 MB free tier can absorb once the model, the image
buffers and the web server are all resident.

| Host | Free RAM | Sleeps? | Verdict |
|---|---|---|---|
| **Hugging Face Spaces** | 16 GB | after ~48 h idle | **Use this** |
| Render free | 512 MB | after 15 min idle | Risky — OOM, plus ~50 s cold start |
| Render paid (Starter) | 512 MB+ | no | Fine if you are paying |

Render's free tier fails us twice over: PyTorch may not fit in 512 MB at all, and a
service that sleeps after 15 minutes means an evaluator opening our link at 9 pm waits
through a cold start, or times out. Spaces has neither problem.

**This is hosting, nothing more.** Our model runs inside our own container, on CPU, from
weights we trained. No external inference API is called. The problem statement explicitly
permits general-purpose services for "authentication, databases, hosting".

---

## Deploy to Hugging Face Spaces

**1. Make sure the model is committed**

`model/infrapulse_model.pt` (~45 MB) must be in the repository. Check `.gitignore` is not
excluding it:

```bash
git add -f model/infrapulse_model.pt
git status          # confirm it is staged
```

**2. Create the Space**

- Sign in at huggingface.co → **New** → **Space**
- Owner: you · Space name: `infrapulse`
- **Space SDK: Docker** → **Blank**
- Visibility: **Public** (evaluators must reach it without logging in)
- Create

**3. Push the code**

```bash
git remote add space https://huggingface.co/spaces/<your-username>/infrapulse
git push space main
```

You will be asked for your username and an access token (huggingface.co/settings/tokens →
New token → **Write**). Paste the token as the password.

> **Leave the Space's own `README.md` alone.** Hugging Face stores the Space
> configuration (`sdk: docker`, `app_port: 7860`) in its frontmatter. If you overwrite it
> with our project README the Space will not build. If you hit a conflict on push, keep
> theirs for `README.md` and yours for everything else.

**4. Set the secrets**

In the Space → **Settings** → **Variables and secrets** → add:

| Name | Value |
|---|---|
| `INFRAPULSE_SECRET` | any long random string |
| `STAFF_PASSWORD` | the staff password you want |

`INFRAPULSE_SECRET` matters: without it the app generates a new signing key on every
restart, which silently logs everyone out mid-demo.

**5. Wait for the build** (5–10 minutes the first time — PyTorch is large). The Space
shows **Running** when it is live.

**6. Verify before you call it done**

- Open `https://<your-username>-infrapulse.hf.space/health` → should report
  `"loaded": true` and list the four classes. If it says `false`, the model file did not
  get committed.
- Open the site **in a private/incognito window**, so you are testing as an evaluator with
  no session: register, submit a photo, confirm a defect and category come back.
- Log in as staff and confirm the complaint is sitting in the right queue.

---

## Fallback: Render

Only if Spaces is unavailable, and preferably on a paid plan.

1. Push to GitHub.
2. render.com → **New** → **Web Service** → connect the repo.
3. Runtime **Docker**. `render.yaml` in the repo sets the health check and generates
   `INFRAPULSE_SECRET` automatically.
4. Set `STAFF_PASSWORD` manually in the dashboard.

On the free plan, expect a cold start of roughly a minute after idling, and watch for
out-of-memory restarts in the logs.

---

## Before the deadline

- [ ] `/health` returns `"loaded": true`
- [ ] Full flow works in a private window on the live URL
- [ ] Staff accounts exist and each sees only its own category
- [ ] URL is public — no login wall, no "space is private"
- [ ] `INFRAPULSE_SECRET` is set, so restarts do not drop sessions
- [ ] Someone else opens the link on their phone and it works

That last one is worth doing. It is the cheapest way to catch a URL that only works
because your browser happens to hold a session.
