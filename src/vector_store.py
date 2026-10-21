"""Qdrant: savol vektorlari saqlanadigan joy.

Bitta nuqta = bitta savol. Nuqta id'si = Postgres'dagi questions.id (ikki baza shu
raqam orqali bog'lanadi). Payload'da faqat bog'lanish va filtr uchun kerakli raqamlar;
savol matni Postgres'da turadi (haqiqat manbai bitta bo'lsin).
"""

from functools import lru_cache

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
