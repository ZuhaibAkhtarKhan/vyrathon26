# Grabpic — Database Schema

All tables are portable between PostgreSQL (production) and SQLite (tests /
local dev). Embedding vectors are stored as JSON-encoded arrays of 128
`float32` values via a SQLAlchemy `TypeDecorator`. When deploying on Postgres
with `pgvector`, swap the decorator for `sqlalchemy_pgvector.Vector(128)` —
no other code changes required.

## Entity-Relationship Diagram

```
  +----------+       +-------------+       +--------+
  |  images  |<----- |    faces    | ----->| grabs  |
  |          |  1..* |             | *..1  |        |
  +----------+       +-------------+       +--------+
       ^                                        ^
       |                                        |
       |          +---------------+             |
       +----------|  image_grabs  |-------------+
                  |  (m:n link)   |
                  +---------------+
```

- `images` has 0..N `faces`.
- `grabs` has 0..N `faces` (all faces in one cluster).
- `image_grabs` is a materialised uniqueness-enforced m:n link derived from
  `faces`. It answers "give me every image containing grab_id X" in a single
  indexed query without walking all face rows.

## `images`

The canonical record of an ingested photo.

| Column          | Type          | Constraints                                       |
|-----------------|---------------|---------------------------------------------------|
| `id`            | `UUID`        | PK, default `gen_random_uuid()`                   |
| `filename`      | `TEXT`        | NOT NULL                                          |
| `storage_path`  | `TEXT`        | NOT NULL, UNIQUE — absolute / bucket path         |
| `content_hash`  | `CHAR(64)`    | NOT NULL, UNIQUE — SHA-256 of raw bytes, dedup    |
| `width`         | `INTEGER`     | NOT NULL                                          |
| `height`        | `INTEGER`     | NOT NULL                                          |
| `face_count`    | `INTEGER`     | NOT NULL DEFAULT 0 — cached count of faces        |
| `created_at`    | `TIMESTAMPTZ` | NOT NULL DEFAULT `now()`                          |

Indexes: `content_hash` (unique), `storage_path` (unique).

## `grabs`

One row per unique discovered identity. `centroid` is the running L2-normalised
mean of every face embedding assigned to this grab.

| Column        | Type          | Constraints                          |
|---------------|---------------|--------------------------------------|
| `id`          | `UUID`        | PK, default `gen_random_uuid()`      |
| `label`       | `TEXT`        | NULLable — human-assignable name     |
| `centroid`    | `JSONB / JSON`| NOT NULL — `float[128]`, \|v\|≈1     |
| `face_count`  | `INTEGER`     | NOT NULL DEFAULT 0                   |
| `created_at`  | `TIMESTAMPTZ` | NOT NULL DEFAULT `now()`             |
| `updated_at`  | `TIMESTAMPTZ` | NOT NULL DEFAULT `now()`             |

## `faces`

One row per face bounding box detected in an image.

| Column                  | Type          | Constraints                            |
|-------------------------|---------------|----------------------------------------|
| `id`                    | `UUID`        | PK                                     |
| `image_id`              | `UUID`        | NOT NULL, FK → `images.id` ON DELETE CASCADE |
| `grab_id`               | `UUID`        | NULLable, FK → `grabs.id` ON DELETE SET NULL |
| `embedding`             | `JSONB / JSON`| NOT NULL — `float[128]`                |
| `bbox_x`                | `INTEGER`     | NOT NULL                               |
| `bbox_y`                | `INTEGER`     | NOT NULL                               |
| `bbox_w`                | `INTEGER`     | NOT NULL                               |
| `bbox_h`                | `INTEGER`     | NOT NULL                               |
| `detection_confidence`  | `FLOAT`       | NOT NULL                               |
| `created_at`            | `TIMESTAMPTZ` | NOT NULL DEFAULT `now()`               |

Indexes: `(grab_id)`, `(image_id)`.

## `image_grabs`

The explicit many-to-many link between photos and identities. Enforces
"a (image, grab) pair appears at most once" and stores how many faces of that
grab appear in the image.

| Column        | Type       | Constraints                              |
|---------------|------------|------------------------------------------|
| `image_id`    | `UUID`     | NOT NULL, FK → `images.id` ON DELETE CASCADE |
| `grab_id`     | `UUID`     | NOT NULL, FK → `grabs.id` ON DELETE CASCADE  |
| `face_count`  | `INTEGER`  | NOT NULL DEFAULT 1                       |
| PK            | composite  | `(image_id, grab_id)`                    |

Indexes: `(grab_id)` for the "fetch my photos" query.

## Query Patterns

1. **Fetch photos for a user (selfie auth returned `grab_id = X`):**
   ```sql
   SELECT i.* FROM images i
   JOIN image_grabs ig ON ig.image_id = i.id
   WHERE ig.grab_id = :X
   ORDER BY i.created_at DESC;
   ```
2. **1:N identity lookup (selfie auth):**
   loads all rows of `grabs`, computes cosine similarity in numpy, returns
   `argmax`.
3. **Detect duplicate uploads:**
   `SELECT id FROM images WHERE content_hash = :sha256`.
