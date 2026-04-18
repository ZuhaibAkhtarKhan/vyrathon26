# Grabpic — Architecture

## 1. Problem Recap

Grabpic is an image indexing + identity engine. At a large event (e.g., a
marathon), photographers push tens of thousands of photos into a storage bucket.
Attendees authenticate with a selfie and, in return, get every photo in which
their face appears.

This means we need four things working together:

1. A **discovery / ingestion pipeline** that crawls storage, detects every face
   in every image, generates a numerical embedding per face, and clusters
   embeddings that belong to the same real person into a stable identity we
   call a `grab_id`.
2. A **durable schema** that preserves the `image → { grab_id, grab_id, ... }`
   many-to-many relationship (one photo can contain many runners; one runner
   appears in many photos).
3. A **selfie authentication endpoint** that performs a 1:N lookup of a
   user-supplied face against all known `grab_id`s.
4. A **retrieval endpoint** that, given a `grab_id`, returns every image where
   that person appears.

## 2. High-level Architecture

```
          +------------------+
          |  Photo Storage   |   (local dir / S3 / GCS / ...)
          |  ./data/images/  |
          +--------+---------+
                   |
                   | crawl + hash (dedup)
                   v
  +----------------+----------------+
  |        Ingestion Pipeline       |
  |                                 |
  |  1. Read image bytes            |
  |  2. Detect faces (YuNet ONNX)   |
  |  3. Align + embed (SFace ONNX)  |
  |  4. Cluster vs. known centroids |
  |  5. Persist image + faces +     |
  |     grab links                  |
  +----------------+----------------+
                   |
                   v
     +-------------+-------------+
     |      PostgreSQL           |
     |                           |
     |  images  grabs  faces     |
     |  image_grabs (m:n link)   |
     +-------------+-------------+
                   ^
                   |
  +----------------+----------------+
  |          FastAPI HTTP           |
  |                                 |
  |  POST /ingest/scan              |
  |  POST /ingest/image             |
  |  POST /auth/selfie    -> grab   |
  |  GET  /grabs/{id}/images        |
  |  GET  /grabs                    |
  |  GET  /images/{id}/download     |
  |  GET  /health                   |
  +---------------------------------+
                   ^
                   |
         Swagger UI + ReDoc + Postman
```

## 3. Component Choices & Rationale

### 3.1 Web framework — FastAPI
- Automatically generates OpenAPI 3 / Swagger UI / ReDoc — ticks the "docs"
  nice-to-have for free.
- Pydantic v2 validation gives strong request/response typing and consistent
  error payloads.
- Starlette's `UploadFile` streams multipart uploads without buffering the
  entire image into memory.

### 3.2 Database — PostgreSQL (SQLite fallback)
- Problem statement lists Postgres as preferred.
- Relational model is a natural fit for the `images`↔`grabs` many-to-many
  relationship.
- SQLAlchemy 2.0 is used so the same code runs on Postgres in production and
  SQLite for tests / local dev with zero config (`DATABASE_URL=sqlite:///./grabpic.db`).
- Embeddings are stored as JSON-encoded `float[128]` for portability. On
  Postgres this can be upgraded to `pgvector` by swapping the `Vector` TypeDecorator; the service layer does all similarity math on numpy arrays loaded
  into memory, which is more than fast enough for the hackathon scale (≤ 50 k
  photos, ≤ 500 runners → ≤ ~150 k face rows, centroid search is 500 x 128
  dot products per call).

### 3.3 Face recognition — OpenCV YuNet + SFace
- Why not `dlib` / `face_recognition`? They require a C++ toolchain on Windows
  and routinely fail to `pip install` cleanly.
- Why not `InsightFace` / `DeepFace`? Heavy (onnxruntime-gpu, TF) and slow cold
  starts.
- **YuNet** (`face_detection_yunet_2023mar.onnx`, ~340 kB) is a modern,
  lightweight detector shipped by the OpenCV zoo.
- **SFace** (`face_recognition_sface_2021dec.onnx`, ~37 MB) produces 128-d
  L2-normalisable embeddings with a published cosine threshold of **0.363**
  for same-identity decisions.
- Both ship as ONNX and are consumed by `cv2.FaceDetectorYN` /
  `cv2.FaceRecognizerSF`, which are in `opencv-contrib-python` — one pip
  package, no native toolchain needed.
- Models auto-download from the OpenCV zoo on first boot into `./models/`.

The `FaceEngine` is abstract (`app/services/face_engine.py`) so a different
backend (InsightFace, a hosted Rekognition API, etc.) can be dropped in
without touching the ingestion / auth code. A `StubFaceEngine` exists for
deterministic tests that must run without downloading model weights.

### 3.4 Clustering / identity assignment
Online single-pass clustering against centroids (sometimes called
"leader-follower"):

```
for each face F in a new image:
    E = embed(F)
    best, score = argmax over grabs G of cosine(E, G.centroid)
    if score >= THRESHOLD:
        assign F to best
        best.centroid = normalize((best.centroid * best.face_count + E) / (best.face_count + 1))
        best.face_count += 1
    else:
        create new grab with centroid = E, face_count = 1
        assign F to new grab
```

This is O(faces × grabs) which is fine at hackathon scale. The incremental
centroid update is a standard running-mean and keeps the cluster centre
stable as more instances of the same person are observed.

For production we would swap the linear scan for a FAISS / pgvector IVF index
and add a batch re-clustering job, but the algorithm is the same.

### 3.5 Selfie authentication
1. Detect all faces in the uploaded selfie.
2. Pick the single largest face (the user's own).
3. Embed it, compare cosine similarity to every grab centroid.
4. If the best score ≥ threshold, return `{grab_id, score, confidence}`.
5. Otherwise return HTTP 404 with `NO_MATCH`.

The response is the `grab_id` itself — it acts as a bearer token for the
retrieval endpoint. In a production system this would be exchanged for a
short-lived signed JWT; that is out of scope for this hackathon but the swap
point is documented in `app/routers/auth.py`.

## 4. Request Lifecycle Example

**User uploads `selfie.jpg` to `/api/v1/auth/selfie`:**

1. FastAPI validates the multipart body (non-empty, MIME allowlist).
2. `AuthService.authenticate_selfie(bytes)`:
   - Decodes with OpenCV.
   - `FaceEngine.detect` → bboxes + alignment landmarks.
   - Picks largest bbox, raises 400 if zero faces, 400 with hint if > 3 faces
     ("is this really a selfie?").
   - `FaceEngine.embed_aligned` → 128-d numpy array.
   - `Matcher.find_best_grab(embedding)` scans all grab centroids.
3. Returns `{grab_id, similarity, confidence}` or 404.

## 5. Error Handling Strategy

A single `register_exception_handlers()` in `app/main.py` maps:

| Exception                    | HTTP | Response code     |
|------------------------------|------|-------------------|
| `NoFaceDetectedError`        | 400  | `NO_FACE`         |
| `MultipleFacesInSelfieError` | 400  | `AMBIGUOUS_SELFIE`|
| `UnsupportedMediaError`      | 415  | `UNSUPPORTED_MEDIA` |
| `NotFoundError`              | 404  | `NOT_FOUND`       |
| `NoMatchError`               | 404  | `NO_MATCH`        |
| `ValidationError` (pydantic) | 422  | `VALIDATION_ERROR`|
| `Exception` (catch-all)      | 500  | `INTERNAL_ERROR`  |

Every error body follows:

```json
{
  "error": {
    "code": "NO_FACE",
    "message": "No face detected in the uploaded image.",
    "details": { ... optional ... }
  }
}
```

## 6. Scaling Notes (out of scope but worth calling out)

- Ingestion is CPU-bound on the ONNX inference. For 50 k photos, fan out with
  a Celery / RQ worker pool; the pipeline is already stateless per-image
  except for the centroid update, which is the only place that needs a lock
  (or better: batched re-clustering).
- Replace the in-process linear centroid scan with pgvector + an HNSW index.
- Store originals in object storage (S3), keep only derived metadata +
  thumbnail paths in Postgres.
- Add a periodic re-clustering job (Chinese Whispers / HDBSCAN) to split
  grabs that drifted and merge duplicates that were created when two people
  look alike in poor lighting.
