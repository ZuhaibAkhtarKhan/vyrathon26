# Deploying Grabpic to Render

The repo ships with a ready-to-use `render.yaml` Blueprint that stands up the
whole stack — web service, Postgres database, and a persistent disk for
uploaded photos + cached ONNX models — in one click.

## What you need

* A [Render account](https://render.com) (free to sign up).
* This repo pushed to GitHub (already done).
* 6 minutes.

## One-click deploy

1. In the Render dashboard: **New +** → **Blueprint**.
2. Connect your GitHub account if you haven't already, then pick the
   `vyrathon26` repository.
3. Render detects `render.yaml` and shows you the resources it will create:
   * `grabpic-api` — web service (Docker build)
   * `grabpic-db` — PostgreSQL database
   * `grabpic-data` — 5 GB persistent disk
4. Click **Apply**. Render starts building.
5. First build takes ~5–6 minutes (it's Dockerizing + pulling
   `opencv-contrib-python` + `numpy`).
6. When the service turns **"Live"** (green), open it. Your API is available at:

   ```
   https://grabpic-api.onrender.com/api/v1/docs
   ```

   (Render assigns a slug based on your service name — yours may differ.)

## After deploy — how to use it

Because there's no local folder on Render for you to drop files into, the
workflow shifts slightly compared to local development:

* ❌ **Don't use** `POST /ingest/scan` — the server has no images to scan.
* ✅ **Use** `POST /ingest/image` to upload photos one at a time. The server
  writes each upload to the persistent disk at `/var/data/images/{sha256}.jpg`.

### Bulk-upload a folder of local photos to the deployed API

PowerShell:

```powershell
$BASE = "https://grabpic-api.onrender.com"
Get-ChildItem C:\path\to\photos\*.jpg | ForEach-Object {
    Write-Host "Uploading $($_.Name)..."
    curl.exe -sS -X POST "$BASE/api/v1/ingest/image" -F "file=@$($_.FullName)"
    Write-Host ""
}
```

Bash / macOS / Linux:

```bash
BASE="https://grabpic-api.onrender.com"
for f in ~/photos/*.jpg; do
  curl -sS -X POST "$BASE/api/v1/ingest/image" -F "file=@$f"
  echo
done
```

### Then test like you did locally

* Swagger UI: `https://<your-slug>.onrender.com/api/v1/docs`
* `POST /api/v1/auth/selfie` — upload your selfie → receive `grab_id`.
* `GET /api/v1/grabs/{grab_id}/images` — fetch every photo that person is in.

## Cost

* **Web service — Starter plan: ~$7/month.** Required because the free plan
  doesn't include persistent disks, and without a disk every uploaded photo
  is lost the next time the container restarts.
* **Postgres — Free plan: $0 for the first 90 days**, then you must upgrade
  or move to another Postgres provider (Neon, Supabase, ElephantSQL). Keep
  backups.
* **Persistent disk:** bundled with the web service plan, configured to
  5 GB in `render.yaml`. Increase `sizeGB` there if you expect more photos.

Total: **~$7/month** for a demo. Free alternatives if cost is a blocker:
see §"Free-tier alternative" below.

## Watching logs / debugging

* In the Render dashboard click into `grabpic-api` → **Logs** tab for live
  stdout. Look for `Booting Grabpic v1.0.0 (face_engine=opencv)` and
  `Uvicorn running on http://0.0.0.0:8000`.
* On the first `/auth` or `/ingest` call you'll see
  `Downloading https://github.com/opencv/opencv_zoo/...` — that's the
  ONNX models landing in `/var/data/models/`. Subsequent restarts reuse
  the cached files.
* Health: `curl https://<your-slug>.onrender.com/api/v1/health` — should
  return `{"status":"ok","face_engine":"opencv","database":"ok"}`.

## Updating the deployed version

`autoDeploy: true` is set in `render.yaml`, so any push to `main` on GitHub
automatically triggers a rebuild. To disable, change it to `false` and use
the "Manual Deploy" button in the dashboard.

## Tweaking configuration after deploy

All environment variables listed in `render.yaml` can be edited live in the
Render dashboard: **grabpic-api** → **Environment** → edit → Save. The
service restarts with the new values. Common knobs:

| Variable                          | Purpose                                                |
|-----------------------------------|--------------------------------------------------------|
| `FACE_MATCH_THRESHOLD`            | Lower (e.g. `0.25`) to reduce `NO_MATCH` false negs.   |
| `FACE_DETECTION_SCORE_THRESHOLD`  | Lower (e.g. `0.4`) if many photos ingest with 0 faces. |
| `MAX_UPLOAD_BYTES`                | Raise if you want to accept > 10 MiB photos.           |

## Free-tier alternative

If you absolutely need zero-cost hosting and can tolerate cold starts +
lost uploads on restart:

1. In `render.yaml`, change `plan: starter` to `plan: free`.
2. Delete the entire `disk:` block.
3. Change the `STORAGE_DIR` and `MODEL_DIR` env vars to `/tmp/images` and
   `/tmp/models` (writable on the free plan).
4. Redeploy.

What you give up: every time the service cold-starts (after 15 min idle,
or on redeploy), all uploaded photos and cached ONNX models are wiped. The
Postgres rows remain but they'll reference paths that no longer exist —
you'd need to re-upload photos to restore retrieval.

For a fully free, persistent alternative, use **Fly.io** — see the
README's deployment section.
