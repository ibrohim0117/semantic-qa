"""Qdrant bilan ishlovchi kod testlari: src/vector_store.py va src/index_questions.py.

Serversiz va modelsiz: Qdrant xotirada ishlaydi (QdrantClient(":memory:")), vektorlar
qo'lda yasaladi — qaysi biri qaysiga yaqinligi oldindan aniq ma'lum.
"""

import numpy as np
import pytest
from qdrant_client import QdrantClient, models

from src.index_questions import QuestionRow, build_points, existing_ids, sync_points
from src.vector_store import VECTOR_SIZE, ensure_collection, search

NAME = "test_questions"

# Xotiradagi Qdrant payload indekslarini qo'llamaydi va bu haqda ogohlantiradi
pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect")


def unit(*positions: int) -> np.ndarray:
    """Berilgan o'rinlarda 1, qolganida 0 bo'lgan, uzunligi 1 ga keltirilgan vektor."""
    vector = np.zeros(VECTOR_SIZE, dtype=np.float32)
    vector[list(positions)] = 1.0
    return vector / np.linalg.norm(vector)


def row(question_id: int, answer_id: int) -> QuestionRow:
    return QuestionRow(question_id, f"savol {question_id}", answer_id, category_id=1, topic_id=2)


@pytest.fixture
def client() -> QdrantClient:
    return QdrantClient(":memory:")


def test_ensure_collection_creates_only_once(client):
    assert ensure_collection(client, NAME) is True
    assert ensure_collection(client, NAME) is False
    assert client.get_collection(NAME).config.params.vectors.size == VECTOR_SIZE


def test_ensure_collection_rejects_wrong_vector_size(client):
    client.create_collection(
        NAME, vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE)
    )
    with pytest.raises(ValueError, match="o'lchami 3"):
        ensure_collection(client, NAME)


def test_recreate_removes_old_points(client):
    ensure_collection(client, NAME)
    sync_points(client, NAME, build_points([row(1, 10)], np.array([unit(0)])))
    assert ensure_collection(client, NAME, recreate=True) is True
    assert client.count(NAME).count == 0


def test_build_points_copies_ids_and_payload():
    (point,) = build_points([row(13, 3)], np.array([unit(0)]))
    assert point.id == 13
    assert point.payload == {"answer_id": 3, "category_id": 1, "topic_id": 2}
    assert len(point.vector) == VECTOR_SIZE


def test_sync_points_writes_new_and_removes_stale(client):
    ensure_collection(client, NAME)
    sync_points(
        client,
        NAME,
        build_points([row(1, 10), row(2, 10), row(3, 20)], np.array([unit(0), unit(1), unit(2)])),
    )
    # 3-savol Postgres'dan o'chirildi, 4-savol qo'shildi
    deleted = sync_points(
        client,
        NAME,
        build_points([row(1, 10), row(2, 10), row(4, 30)], np.array([unit(0), unit(1), unit(3)])),
    )
    assert deleted == 1
    assert existing_ids(client, NAME) == {1, 2, 4}


def test_sync_points_twice_creates_no_duplicates(client):
    ensure_collection(client, NAME)
    points = build_points([row(1, 10), row(2, 20)], np.array([unit(0), unit(1)]))
    sync_points(client, NAME, points)
    assert sync_points(client, NAME, points) == 0
    assert client.count(NAME).count == 2


def test_existing_ids_reads_all_pages(client):
    ensure_collection(client, NAME)
    rows = [row(i, i) for i in range(1, 1501)]  # scroll sahifasi 1000 ta, demak 2 sahifa
    vectors = np.array([unit(i % VECTOR_SIZE) for i in range(1500)])
    sync_points(client, NAME, build_points(rows, vectors))
    assert existing_ids(client, NAME) == set(range(1, 1501))


def test_search_returns_nearest_first(client):
    ensure_collection(client, NAME)
    sync_points(
        client,
        NAME,
        build_points(
            [row(1, 10), row(2, 20), row(3, 30)], np.array([unit(0), unit(0, 1), unit(5)])
        ),
    )
    hits = search(client, NAME, unit(0), limit=3)
    assert [h.question_id for h in hits] == [1, 2, 3]  # aynan bir xil, qisman o'xshash, aloqasiz
    assert [h.answer_id for h in hits] == [10, 20, 30]
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert hits[0].score > hits[1].score > hits[2].score
