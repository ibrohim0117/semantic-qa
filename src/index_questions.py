"""Savollarni indekslash: Postgres -> bge-m3 -> Qdrant.

Postgres haqiqat manbai, Qdrant undan yasalgan qidiruv ko'rsatkichi (har doim shu yo'nalishda).
Idempotent: nuqta id = savol id (upsert), Postgres'da yo'q nuqtalar Qdrant'dan o'chiriladi.

Ishga tushirish:
    poetry run python -m src.index_questions             # yangilash
    poetry run python -m src.index_questions --recreate  # noldan (masalan, model almashganda)
"""

import argparse
import asyncio
import time
from dataclasses import dataclass

import numpy as np
from qdrant_client import QdrantClient, models
from sqlalchemy import select

from src.config import get_settings
from src.db.models import Answer, Question
from src.db.session import get_engine, get_sessionmaker
from src.embeddings import get_embedder
from src.vector_store import ensure_collection, get_qdrant

UPSERT_BATCH = 256  # bitta so'rovda Qdrant'ga yuboriladigan nuqtalar (~1 MB)


@dataclass(frozen=True)
class QuestionRow:
    id: int
    text: str
    answer_id: int
    category_id: int
    topic_id: int


async def fetch_questions() -> list[QuestionRow]:
    """Faol savollar va ularning javobidagi kategoriya/mavzu (faqat faol javoblar)."""
    stmt = (
        select(Question.id, Question.text, Question.answer_id, Answer.category_id, Answer.topic_id)
        .join(Answer, Answer.id == Question.answer_id)
        .where(Question.is_active, Answer.status == "active")
        .order_by(Question.id)
    )
    async with get_sessionmaker()() as session:
        rows = (await session.execute(stmt)).all()
    await get_engine().dispose()
    return [QuestionRow(*row) for row in rows]


def build_points(rows: list[QuestionRow], vectors: np.ndarray) -> list[models.PointStruct]:
    return [
        models.PointStruct(
            id=row.id,
            vector=vector.tolist(),
            payload={
                "answer_id": row.answer_id,
                "category_id": row.category_id,
                "topic_id": row.topic_id,
            },
        )
        for row, vector in zip(rows, vectors, strict=True)
    ]


def existing_ids(client: QdrantClient, name: str) -> set[int]:
    """Kolleksiyadagi barcha nuqta id'lari (vektor va payload'siz, tez)."""
    ids: set[int] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            name, limit=1000, offset=offset, with_payload=False, with_vectors=False
        )
        ids.update(int(p.id) for p in points)
        if offset is None:
            return ids


def sync_points(client: QdrantClient, name: str, points: list[models.PointStruct]) -> int:
    """Nuqtalarni yozadi, ro'yxatda yo'q eski nuqtalarni o'chiradi.

    O'chirilgan nuqtalar sonini qaytaradi.
    """
    for start in range(0, len(points), UPSERT_BATCH):
        client.upsert(name, points=points[start : start + UPSERT_BATCH], wait=True)
    stale = existing_ids(client, name) - {int(p.id) for p in points}
    if stale:
        client.delete(name, points_selector=models.PointIdsList(points=sorted(stale)), wait=True)
    return len(stale)


def main(recreate: bool) -> None:
    settings = get_settings()
    client = get_qdrant()
    name = settings.qdrant_collection
    created = ensure_collection(client, name, recreate=recreate)
    print(f"Kolleksiya '{name}': {'yangi yaratildi' if created else 'bor edi'}")

    rows = asyncio.run(fetch_questions())
    print(f"Postgres'dan olindi: {len(rows)} ta faol savol")

    t0 = time.perf_counter()
    vectors = get_embedder().encode([row.text for row in rows])
    print(f"Vektorlandi: {vectors.shape} — {time.perf_counter() - t0:.1f} s")

    t0 = time.perf_counter()
    deleted = sync_points(client, name, build_points(rows, vectors))
    count = client.count(name, exact=True).count
    print(
        f"Qdrant'ga yozildi: {time.perf_counter() - t0:.1f} s, eski nuqtalar o'chirildi: {deleted}"
    )
    print(
        f"Tekshiruv: Qdrant'da {count} nuqta, Postgres'da {len(rows)} savol — "
        f"{'mos ✅' if count == len(rows) else 'MOS EMAS ❌'}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Savollarni Qdrant'ga indekslash")
    parser.add_argument("--recreate", action="store_true", help="kolleksiyani noldan qurish")
    main(parser.parse_args().recreate)
