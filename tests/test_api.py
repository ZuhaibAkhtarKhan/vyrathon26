"""End-to-end API tests via FastAPI's TestClient."""

from __future__ import annotations

from tests.conftest import make_embedding, make_face, make_image_bytes


def test_health_endpoint(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_root_redirect_info(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "Grabpic"
    assert body["docs"].endswith("/docs")


def test_ingest_single_image(client, face_engine):
    data = make_image_bytes((70, 70, 70))
    face_engine.program_for_bytes(data, [make_face(make_embedding(seed=1))])

    r = client.post(
        "/api/v1/ingest/image",
        files={"file": ("a.png", data, "image/png")},
    )
    assert r.status_code == 200, r.text
    body = r.json()["result"]
    assert body["faces_detected"] == 1
    assert len(body["grab_ids"]) == 1
    assert body["is_new_image"] is True


def test_ingest_rejects_unsupported_mime(client):
    r = client.post(
        "/api/v1/ingest/image",
        files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")},
    )
    assert r.status_code == 415
    assert r.json()["error"]["code"] == "UNSUPPORTED_MEDIA"


def test_ingest_rejects_corrupt_image(client):
    # Pass application/octet-stream so MIME check lets it through, but bytes
    # are not a valid image — decode should fail.
    r = client.post(
        "/api/v1/ingest/image",
        files={"file": ("not-really.png", b"not an image", "image/png")},
    )
    assert r.status_code == 415
    assert r.json()["error"]["code"] == "UNSUPPORTED_MEDIA"


def test_full_flow_ingest_then_selfie_auth_then_retrieve(client, face_engine):
    # --- Setup: ingest two photos of person A, one of person B. ---
    emb_a = make_embedding(seed=1)
    emb_b = make_embedding(seed=7)

    bytes_a1 = make_image_bytes((10, 10, 10))
    bytes_a2 = make_image_bytes((20, 20, 20))
    bytes_b = make_image_bytes((200, 100, 50))

    face_engine.program_for_bytes(bytes_a1, [make_face(emb_a)])
    face_engine.program_for_bytes(bytes_a2, [make_face(emb_a)])
    face_engine.program_for_bytes(bytes_b, [make_face(emb_b)])

    for name, data in [
        ("a1.png", bytes_a1),
        ("a2.png", bytes_a2),
        ("b.png", bytes_b),
    ]:
        r = client.post(
            "/api/v1/ingest/image",
            files={"file": (name, data, "image/png")},
        )
        assert r.status_code == 200, r.text

    # --- Act: authenticate with a selfie that has embedding ~ emb_a. ---
    selfie_bytes = make_image_bytes((55, 33, 77))
    face_engine.program_for_bytes(selfie_bytes, [make_face(emb_a)])

    r = client.post(
        "/api/v1/auth/selfie",
        files={"file": ("selfie.png", selfie_bytes, "image/png")},
    )
    assert r.status_code == 200, r.text
    auth_body = r.json()
    grab_id = auth_body["grab_id"]
    assert auth_body["similarity"] > 0.9
    assert auth_body["confidence"] in {"medium", "high"}

    # --- Assert: /grabs/{id}/images returns exactly the two A photos. ---
    r = client.get(f"/api/v1/grabs/{grab_id}/images")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert {img["filename"] for img in body["images"]} == {"a1.png", "a2.png"}

    # And the B photo is linked to a different grab.
    r = client.get("/api/v1/grabs")
    assert r.status_code == 200
    all_grabs = r.json()
    assert len(all_grabs) == 2


def test_selfie_returns_404_when_no_grabs_known(client, face_engine):
    selfie = make_image_bytes((1, 2, 3))
    face_engine.program_for_bytes(selfie, [make_face(make_embedding(seed=1))])

    r = client.post(
        "/api/v1/auth/selfie",
        files={"file": ("selfie.png", selfie, "image/png")},
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "NO_MATCH"


def test_selfie_returns_400_when_no_face_detected(client, face_engine):
    selfie = make_image_bytes((1, 2, 3))
    # Don't program → scripted engine returns zero faces.
    r = client.post(
        "/api/v1/auth/selfie",
        files={"file": ("selfie.png", selfie, "image/png")},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "NO_FACE"


def test_grab_images_404s_on_unknown_id(client):
    r = client.get("/api/v1/grabs/does-not-exist/images")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "NOT_FOUND"


def test_multiple_faces_in_one_image_links_to_multiple_grabs(client, face_engine):
    group = make_image_bytes((100, 200, 50))
    face_engine.program_for_bytes(
        group,
        [
            make_face(make_embedding(seed=1), bbox=(0, 0, 10, 10)),
            make_face(make_embedding(seed=2), bbox=(10, 0, 10, 10)),
        ],
    )
    r = client.post(
        "/api/v1/ingest/image",
        files={"file": ("group.png", group, "image/png")},
    )
    assert r.status_code == 200
    body = r.json()["result"]
    assert body["faces_detected"] == 2
    assert len(body["grab_ids"]) == 2

    image_id = body["image_id"]
    r = client.get(f"/api/v1/images/{image_id}")
    assert r.status_code == 200
    detail = r.json()
    assert detail["face_count"] == 2
    assert len(detail["faces"]) == 2
    assert len(detail["grab_ids"]) == 2


def test_scan_directory_endpoint(client, face_engine, tmp_path):
    folder = tmp_path / "scan"
    folder.mkdir()

    for i, color in enumerate([(10, 0, 0), (0, 10, 0), (0, 0, 10)]):
        data = make_image_bytes(color)
        (folder / f"p{i}.png").write_bytes(data)
        face_engine.program_for_bytes(data, [make_face(make_embedding(seed=i + 1))])

    r = client.post(
        "/api/v1/ingest/scan",
        json={"directory": str(folder), "recursive": True},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["scanned"] == 3
    assert body["ingested"] == 3
    assert body["total_faces"] == 3
    assert body["new_grabs_created"] == 3


def test_scan_directory_404s_on_missing_path(client):
    r = client.post(
        "/api/v1/ingest/scan",
        json={"directory": "/definitely/does/not/exist/xyz123"},
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "NOT_FOUND"


def test_openapi_schema_is_served(client):
    r = client.get("/api/v1/openapi.json")
    assert r.status_code == 200
    schema = r.json()
    assert schema["info"]["title"].startswith("Grabpic")
    # Spot-check a few routes are documented.
    paths = schema["paths"]
    assert "/api/v1/auth/selfie" in paths
    assert "/api/v1/ingest/scan" in paths
    assert "/api/v1/grabs/{grab_id}/images" in paths
