# Grabpic — Intelligent Identity & Retrieval Engine

> Vyrathon '26 — Backend problem set

Grabpic is a backend service for large-scale event photography. Photographers
dump tens of thousands of photos into storage; attendees authenticate with a
selfie and, in return, receive every photo they appear in. The system uses
facial recognition to assign each unique face a stable `grab_id` and persists
the full many-to-many relationship between images and identities.

- [Features](#features) • [Quickstart](#quickstart) • [API](#api-reference) •
  [cURL examples](#curl-examples) • [Architecture](#architecture) •
  [Tests](#tests) • [Configuration](#configuration)

---

## Features

- **Discovery & transformation**
  - Crawl a directory (or ingest single uploads) → detect faces → generate
    embeddings → cluster into `grab_id`s → persist.
  - Many-to-many `images ↔ grabs` mapping, plus a full `faces` table with
    bounding boxes + confidences for every detection.
  - Content-hash deduplication — ingesting the same image twice is a no-op.
- **Selfie-as-a-key authentication**
  - Upload a selfie, get back the best-matching `grab_id`.
  - Cosine-similarity lookup with a tunable threshold (default 0.363, the
    published SFace threshold).
  - Clear error codes for `NO_FACE`, `NO_MATCH`, `UNSUPPORTED_MEDIA`, etc.
- **Data extraction**
  - `GET /grabs/{grab_id}/images` returns every photo a person appears in.
  - `GET /images/{image_id}` returns metadata + all detected faces.
  - `GET /images/{image_id}/download` streams the original file.
- **Production-shaped**
  - Auto-generated **Swagger UI** at `/api/v1/docs` and **ReDoc** at
    `/api/v1/redoc`.
  - Postman collection in [`docs/grabpic.postman_collection.json`](docs/grabpic.postman_collection.json).
  - Consistent `{"error": {"code","message","details"}}` payload on every
    failure.
  - 31 unit + integration tests (`pytest`).
  - Dockerfile + docker-compose wiring Postgres.

## Quickstart

### 1. Local (Python 3.11+, SQLite, no Docker)

```bash
git clone <this-repo>
cd vyrathon26
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS / Linux
# source .venv/bin/activate

pip install --only-binary :all: -r requirements.txt
cp .env.example .env    # (Windows: Copy-Item .env.example .env)

uvicorn app.main:app --reload
```

Open:
- Swagger UI: <http://localhost:8000/api/v1/docs>
- ReDoc: <http://localhost:8000/api/v1/redoc>
- Health: <http://localhost:8000/api/v1/health>

On first real ingest/auth call the ~38 MB YuNet + SFace ONNX models are
auto-downloaded into `./models/`. If your network is restricted, set
`FACE_ENGINE=stub` in `.env` to run without model weights (recognition
becomes a deterministic hash of pixel bytes, good enough for wiring tests
but not for real face matching).

### 2. Docker (with Postgres)

```bash
docker compose up --build
```

This starts Postgres 16 and the API on <http://localhost:8000>. Photos in
`./data/images` on your host are mounted into the API container at
`/app/data/images` and can be indexed via `POST /ingest/scan`.

### 3. Deploy to Render (one-click)

The repo ships with a `render.yaml` Blueprint. In the Render dashboard click
**New + → Blueprint**, pick this repo, **Apply**, and in ~6 minutes your API
is live with a managed Postgres + persistent disk. Full guide:
[`docs/DEPLOY.md`](docs/DEPLOY.md).

## API Reference

| Method | Path                                | Purpose                                                                 |
|--------|-------------------------------------|-------------------------------------------------------------------------|
| GET    | `/api/v1/health`                    | Liveness + DB + face-engine status.                                     |
| POST   | `/api/v1/ingest/scan`               | Crawl a directory and index every image it finds.                       |
| POST   | `/api/v1/ingest/image`              | Ingest a single uploaded image (multipart `file`).                      |
| POST   | `/api/v1/auth/selfie`               | Selfie → best-matching `grab_id` (multipart `file`).                    |
| GET    | `/api/v1/grabs`                     | Paginated list of all discovered identities.                            |
| GET    | `/api/v1/grabs/{grab_id}`           | Single identity by id.                                                  |
| GET    | `/api/v1/grabs/{grab_id}/images`    | **Data extraction endpoint** — every image this identity appears in.    |
| GET    | `/api/v1/grabs/{grab_id}/faces`     | Face instances belonging to an identity (debug).                        |
| GET    | `/api/v1/images`                    | Paginated list of ingested images.                                      |
| GET    | `/api/v1/images/{image_id}`         | Image metadata + faces + grab ids.                                      |
| GET    | `/api/v1/images/{image_id}/download`| Stream original bytes.                                                  |

All endpoints are prefixed with `API_PREFIX` (default `/api/v1`).

### Error payload

Every 4xx / 5xx response returns:

```json
{
  "error": {
    "code": "NO_MATCH",
    "message": "No matching identity found...",
    "details": { "faces_in_selfie": 1 }
  }
}
```

| HTTP | Code                | When                                           |
|------|---------------------|------------------------------------------------|
| 400  | `NO_FACE`           | No face detectable in the uploaded selfie.     |
| 400  | `VALIDATION_ERROR`  | Request payload invalid.                       |
| 413  | `PAYLOAD_TOO_LARGE` | Upload exceeds `MAX_UPLOAD_BYTES`.             |
| 415  | `UNSUPPORTED_MEDIA` | Unknown MIME type or undecodable image bytes.  |
| 404  | `NO_MATCH`          | Selfie didn't match any known `grab_id`.       |
| 404  | `NOT_FOUND`         | Unknown image/grab/directory.                  |
| 422  | `VALIDATION_ERROR`  | Pydantic request-body error.                   |
| 500  | `INTERNAL_ERROR`    | Any other unhandled exception.                 |

## cURL examples

```bash
# 1. Health
curl -sS http://localhost:8000/api/v1/health | jq

# 2. Ingest a directory (crawls the server-side STORAGE_DIR by default)
curl -sS -X POST http://localhost:8000/api/v1/ingest/scan \
  -H "Content-Type: application/json" \
  -d '{"directory": "./data/images", "recursive": true}' | jq

# 3. Ingest a single photo
curl -sS -X POST http://localhost:8000/api/v1/ingest/image \
  -F "file=@./photos/race_0001.jpg" | jq

# 4. Authenticate with a selfie
curl -sS -X POST http://localhost:8000/api/v1/auth/selfie \
  -F "file=@./selfie.jpg" | jq
# -> {"grab_id":"...","similarity":0.71,"confidence":"high", ...}

# 5. Fetch all photos for that grab_id (the "Selfie-as-a-key" payoff)
GRAB_ID="PASTE_THE_GRAB_ID_HERE"
curl -sS "http://localhost:8000/api/v1/grabs/$GRAB_ID/images" | jq

# 6. Download a specific image
curl -sSOJ "http://localhost:8000/api/v1/images/<image_id>/download"

# 7. Browse raw catalogs
curl -sS "http://localhost:8000/api/v1/grabs?limit=20" | jq
curl -sS "http://localhost:8000/api/v1/images?limit=20" | jq
```

## Architecture

Full write-up in [`docs/architecture.md`](docs/architecture.md). Schema in
[`docs/schema.md`](docs/schema.md). Highlights:

```
Photos -> Ingestion pipeline -> Postgres/SQLite -> FastAPI -> Swagger/cURL
            |   |   |   |
            |   |   |   +-- Cluster against grab centroids (cosine)
            |   |   +------ SFace  128-d embedding  (ONNX)
            |   +---------- YuNet  face detection   (ONNX)
            +-------------- SHA-256 dedup
```

- **Face engine.** Abstract `FaceEngine` with two implementations:
  - `OpenCVFaceEngine` — `opencv-contrib-python` + YuNet + SFace ONNX
    (auto-downloaded into `./models/`). Production / recommended.
  - `StubFaceEngine` — deterministic hash-of-pixels embedding, used when
    `FACE_ENGINE=stub`. Lets the API boot without model weights, used by
    tests and in restricted-network environments.
- **Matcher.** Online single-pass clustering. Each new face is compared to
  every known grab centroid; above threshold → assign and update centroid
  (running mean, re-normalised); otherwise a new grab is created.
- **Database.** SQLAlchemy 2.0 with a `JSONEncodedVector` `TypeDecorator`
  that stores `float[128]` as JSON text. Swappable for `pgvector` in one
  line.

## Tests

```bash
.\.venv\Scripts\python.exe -m pytest -v    # Windows
# or: pytest -v
```

31 tests covering: utility functions, the clustering algorithm, the full
ingestion pipeline (single face, multi face, idempotency, clustering across
images, directory scans), error handling (unsupported media, missing
directories, unknown grab ids), and the full HTTP round-trip
`ingest → auth → retrieve`.

Tests never hit the network — they swap in the `ScriptedFaceEngine` fixture
which returns pre-programmed faces per decoded image.

## Configuration

All via env vars / `.env` (see [`.env.example`](.env.example)).

| Variable                          | Default                    | Purpose                                                   |
|-----------------------------------|----------------------------|-----------------------------------------------------------|
| `DATABASE_URL`                    | `sqlite:///./grabpic.db`   | SQLAlchemy URL. Postgres is recommended for real use.     |
| `STORAGE_DIR`                     | `./data/images`            | Directory crawled by `/ingest/scan` and where uploads land.|
| `MODEL_DIR`                       | `./models`                 | Where ONNX weights are cached.                            |
| `FACE_ENGINE`                     | `opencv`                   | `opencv` or `stub`.                                       |
| `FACE_MATCH_THRESHOLD`            | `0.363`                    | Cosine threshold for "same identity" (SFace default).     |
| `FACE_DETECTION_SCORE_THRESHOLD`  | `0.6`                      | YuNet detection confidence cutoff.                        |
| `API_PREFIX`                      | `/api/v1`                  | Path prefix for every endpoint.                           |
| `MAX_UPLOAD_BYTES`                | `10485760` (10 MiB)        | Hard cap on any upload.                                   |

## Project layout

```
app/
  main.py              # FastAPI app factory + exception handlers
  config.py            # pydantic-settings
  database.py          # Engine, session, Base, init_db()
  models.py            # ORM: Image, Grab, Face, ImageGrab
  schemas.py           # Pydantic request/response types
  deps.py              # Upload validation dependency
  routers/             # FastAPI routers: ingest, auth, grabs, images, health
  services/
    face_engine.py     # FaceEngine ABC + OpenCV/stub impls
    ingestion.py       # Crawl -> detect -> embed -> cluster -> persist
    matcher.py         # assign_or_create_grab, find_best_grab
    auth.py            # Selfie authentication service
    retrieval.py       # Read queries
    storage.py         # Image decoding + file enumeration
  utils/
    errors.py          # Domain exceptions mapped to HTTP codes
    hashing.py         # sha256 helpers
tests/                 # 31 unit + integration tests
docs/                  # architecture.md, schema.md, Postman collection
```

## License

MIT — built for the Vyrathon '26 hackathon.
