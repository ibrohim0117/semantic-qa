"""Qdrant: savol vektorlari saqlanadigan joy.

Bitta nuqta = bitta savol. Nuqta id'si = Postgres'dagi questions.id (ikki baza shu
raqam orqali bog'lanadi). Payload'da faqat bog'lanish va filtr uchun kerakli raqamlar;
savol matni Postgres'da turadi (haqiqat manbai bitta bo'lsin).
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from qdrant_client import QdrantClient, models

from src.config import get_settings

VECTOR_SIZE = 1024  # bge-m3 vektor o'lchami
INDEXED_FIELDS = ("answer_id", "category_id")  # filtr va guruhlash tez ishlashi uchun


@lru_cache
def get_qdrant() -> QdrantClient:
    return QdrantClient(url=get_settings().qdrant_url)


def ensure_collection(client: QdrantClient, name: str, recreate: bool = False) -> bool:
    """Kolleksiya yo'q bo'lsa yaratadi. Yaratilgan bo'lsa True qaytaradi.

    recreate=True: eskisini o'chirib, noldan yaratadi (masalan, model almashtirilganda).
    """
    if recreate and client.collection_exists(name):
        client.delete_collection(name)

    if client.collection_exists(name):
        size = client.get_collection(name).config.params.vectors.size
        if size != VECTOR_SIZE:
            raise ValueError(
                f"'{name}' kolleksiyasidagi vektor o'lchami {size}, kerak {VECTOR_SIZE}. "
                "Qayta yaratish kerak (recreate=True)."
            )
        return False

    client.create_collection(
        name,
        vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
    )
    for field in INDEXED_FIELDS:
        client.create_payload_index(
            name, field_name=field, field_schema=models.PayloadSchemaType.INTEGER
        )
    return True


@dataclass(frozen=True)
class Hit:
    question_id: int  # Postgres'dagi questions.id
    answer_id: int
    score: float  # cosine o'xshashlik


def search(
    client: QdrantClient, name: str, vector: np.ndarray, limit: int = 10, exact: bool = False
) -> list[Hit]:
    """Eng yaqin savollar (ballari kamayish tartibida).

    exact=True: indeksdan foydalanmay hamma vektorni tekshiradi (taqqoslash uchun).
    """
    response = client.query_points(
        name,
        query=vector.tolist(),
        limit=limit,
        with_payload=["answer_id"],
        search_params=models.SearchParams(exact=exact),
    )
    return [Hit(int(p.id), int(p.payload["answer_id"]), p.score) for p in response.points]


@dataclass(frozen=True)
class AnswerHit:
    answer_id: int
    score: float  # shu javobning eng yaqin savoli bali
    question_id: int  # o'sha eng yaqin savol (foydalanuvchiga "nimaga o'xshadi" deb ko'rsatiladi)


def search_answers(
    client: QdrantClient, name: str, vector: np.ndarray, limit: int = 3
) -> list[AnswerHit]:
    """Eng yaqin `limit` ta TURLI javob, ballari kamayish tartibida.

    Qdrant natijalarni answer_id bo'yicha o'zi guruhlaydi va har guruhdan eng yaqin
    bitta savolni qaytaradi — bitta javobning variantlari ro'yxatni to'ldirib qo'ymaydi.
    """
    response = client.query_points_groups(
        name,
        query=vector.tolist(),
        group_by="answer_id",
        limit=limit,
        group_size=1,
        with_payload=False,
    )
    return [AnswerHit(int(g.id), g.hits[0].score, int(g.hits[0].id)) for g in response.groups]
