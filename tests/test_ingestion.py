"""Integration tests for the ingestion pipeline at the service layer."""

from __future__ import annotations

from app.models import Face, Grab, Image, ImageGrab
from app.services.ingestion import IngestionService
from tests.conftest import make_embedding, make_face, make_image_bytes


def test_ingest_single_face_creates_one_grab(db_session, face_engine):
    data = make_image_bytes((10, 200, 50))
    face_engine.program_for_bytes(data, [make_face(make_embedding(seed=1))])

    service = IngestionService(face_engine=face_engine)
    outcome = service.ingest_bytes(db_session, data=data, filename="a.png")
    db_session.commit()

    assert outcome.is_new_image is True
    assert outcome.faces_detected == 1
    assert len(outcome.grab_ids) == 1
    assert outcome.new_grabs_created == 1

    assert db_session.query(Image).count() == 1
    assert db_session.query(Face).count() == 1
    assert db_session.query(Grab).count() == 1
    assert db_session.query(ImageGrab).count() == 1


def test_ingest_multi_face_image_creates_many_links(db_session, face_engine):
    data = make_image_bytes((30, 30, 30))
    face_engine.program_for_bytes(
        data,
        [
            make_face(make_embedding(seed=1), bbox=(0, 0, 10, 10)),
            make_face(make_embedding(seed=2), bbox=(10, 0, 10, 10)),
            make_face(make_embedding(seed=3), bbox=(20, 0, 10, 10)),
        ],
    )

    service = IngestionService(face_engine=face_engine)
    outcome = service.ingest_bytes(db_session, data=data, filename="group.png")
    db_session.commit()

    assert outcome.faces_detected == 3
    assert len(set(outcome.grab_ids)) == 3
    assert db_session.query(Grab).count() == 3
    assert db_session.query(ImageGrab).count() == 3


def test_ingest_is_idempotent_on_content_hash(db_session, face_engine):
    data = make_image_bytes((200, 200, 0))
    face_engine.program_for_bytes(data, [make_face(make_embedding(seed=5))])

    service = IngestionService(face_engine=face_engine)
    first = service.ingest_bytes(db_session, data=data, filename="dup.png")
    second = service.ingest_bytes(db_session, data=data, filename="dup.png")
    db_session.commit()

    assert first.is_new_image is True
    assert second.is_new_image is False
    assert first.image_id == second.image_id
    assert db_session.query(Image).count() == 1
    assert db_session.query(Face).count() == 1


def test_same_person_across_two_images_clusters_into_one_grab(db_session, face_engine):
    shared = make_embedding(seed=42)

    data_a = make_image_bytes((255, 0, 0))
    data_b = make_image_bytes((0, 255, 0))
    face_engine.program_for_bytes(data_a, [make_face(shared)])
    face_engine.program_for_bytes(data_b, [make_face(shared)])

    service = IngestionService(face_engine=face_engine)
    a = service.ingest_bytes(db_session, data=data_a, filename="a.png")
    b = service.ingest_bytes(db_session, data=data_b, filename="b.png")
    db_session.commit()

    assert a.grab_ids == b.grab_ids
    assert db_session.query(Grab).count() == 1
    assert db_session.query(Image).count() == 2
    assert db_session.query(ImageGrab).count() == 2


def test_scan_directory_crawls_supported_files(db_session, face_engine, tmp_path):
    folder = tmp_path / "photos"
    folder.mkdir()

    pics = {
        "p1.png": (make_embedding(seed=1), (255, 0, 0)),
        "p2.png": (make_embedding(seed=1), (0, 255, 0)),
        "p3.png": (make_embedding(seed=9), (0, 0, 255)),
    }
    for name, (emb, color) in pics.items():
        data = make_image_bytes(color)
        (folder / name).write_bytes(data)
        face_engine.program_for_bytes(data, [make_face(emb)])

    # Distractor non-image file.
    (folder / "notes.txt").write_text("hello")

    service = IngestionService(face_engine=face_engine)
    summary = service.scan_directory(db_session, folder)

    assert summary.scanned == 3
    assert summary.ingested == 3
    assert summary.total_faces == 3
    assert summary.new_grabs_created == 2
    assert db_session.query(Grab).count() == 2
    assert db_session.query(Image).count() == 3
