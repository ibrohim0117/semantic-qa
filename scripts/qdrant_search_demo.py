"""4.3-qadam: Qdrant orqali qidiruv, HNSW va aniq qidiruvni solishtirish, tezlik.

1) Bir nechta savol: Qdrant id va ballarni beradi, Postgres esa ularning matnini.
2) Vaqtinchalik nusxa kolleksiyada HNSW majburan quriladi va natijasi aniq qidiruv
   bilan solishtiriladi (B va C test savollarida). Nusxa oxirida o'chiriladi.
3) Tezlik: savolni vektorlash va Qdrant qidiruvi.

Ishga tushirish: poetry run python -m scripts.qdrant_search_demo
"""

import asyncio
import statistics
import time

import pandas as pd
from qdrant_client import models
from sqlalchemy import select

from src.config import get_settings
from src.db.models import Question
from src.db.session import get_engine, get_sessionmaker
from src.embeddings import get_embedder
from src.vector_store import VECTOR_SIZE, get_qdrant, search

EXAMPLES = [
    "hozir menda yashil pasport bor, uni qachon yangilashim kerak?",
    "ID karta yo'qotib qo'ydim nima qilay",
    "gaz hisoblagichini qanday almashtiraman?",
]
TEMP = "questions_hnsw_check"


async def question_texts(ids: list[int]) -> dict[int, str]:
    async with get_sessionmaker()() as session:
        rows = await session.execute(select(Question.id, Question.text).where(Question.id.in_(ids)))
    return dict(rows.all())


async def show_examples() -> None:
    client, name, embedder = get_qdrant(), get_settings().qdrant_collection, get_embedder()
    print("1) QDRANT ORQALI QIDIRUV (Qdrant: id + ball, Postgres: matn)")
    for query in EXAMPLES:
        hits = search(client, name, embedder.encode_one(query), limit=5)
        texts = await question_texts([h.question_id for h in hits])
        print(f"\n   «{query}»")
        for h in hits:
            print(f"     {h.score:.3f}  [javob {h.answer_id:>3}]  {texts[h.question_id]}")
    await get_engine().dispose()


def copy_with_hnsw() -> None:
    """Asosiy kolleksiyani HNSW majburan quriladigan vaqtinchalik nusxaga ko'chiradi."""
    client, name = get_qdrant(), get_settings().qdrant_collection
    if client.collection_exists(TEMP):
        client.delete_collection(TEMP)
    client.create_collection(
        TEMP,
        vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        # Juda kichik chegara: har qanday segment uchun HNSW quriladi
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=10),
    )
    offset = None
    while True:
        points, offset = client.scroll(name, limit=500, offset=offset, with_vectors=True)
        client.upsert(
            TEMP,
            points=[
                models.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points
            ],
        )
        if offset is None:
            break
    while True:  # indeks qurilishini kutish
        info = client.get_collection(TEMP)
        if (
            info.status == models.CollectionStatus.GREEN
            and info.indexed_vectors_count == info.points_count
        ):
            print(f"\n2) HNSW nusxada qurildi: {info.indexed_vectors_count} ta vektor indeksda")
            return
        time.sleep(0.5)


def compare_hnsw_vs_exact() -> None:
    client, embedder = get_qdrant(), get_embedder()
    questions = pd.concat(
        [pd.read_csv("eval/test_b.csv")["question"], pd.read_csv("eval/test_c.csv")["question"]]
    ).tolist()
    vectors = embedder.encode(questions)

    overlaps, top1_same, max_diff = [], 0, 0.0
    exact_ms, hnsw_ms = [], []
    for vec in vectors:
        t0 = time.perf_counter()
        exact = search(client, TEMP, vec, limit=10, exact=True)
        exact_ms.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        hnsw = search(client, TEMP, vec, limit=10)
        hnsw_ms.append((time.perf_counter() - t0) * 1000)

        overlaps.append(len({h.question_id for h in exact} & {h.question_id for h in hnsw}) / 10)
        top1_same += exact[0].question_id == hnsw[0].question_id
        max_diff = max(max_diff, abs(exact[0].score - hnsw[0].score))

    n = len(vectors)
    print(f"   {n} ta test savolida (B + C), eng yaqin 10 ta:")
    print(f"   HNSW topganining aniq qidiruvga mosligi: {statistics.mean(overlaps):.1%}")
    print(f"   1-o'rin bir xil: {top1_same}/{n}, 1-o'rin ballidagi eng katta farq: {max_diff:.4f}")
    print("\n3) TEZLIK (median)")
    print(
        f"   Qdrant aniq qidiruv: {statistics.median(exact_ms):.1f} ms, "
        f"HNSW: {statistics.median(hnsw_ms):.1f} ms"
    )
    t0 = time.perf_counter()
    for q in questions[:30]:
        embedder.encode_one(q)
    print(f"   Bitta savolni vektorlash: {(time.perf_counter() - t0) / 30 * 1000:.0f} ms")
    client.delete_collection(TEMP)
    print(f"   Vaqtinchalik '{TEMP}' kolleksiyasi o'chirildi")


if __name__ == "__main__":
    asyncio.run(show_examples())
    copy_with_hnsw()
    compare_hnsw_vs_exact()
