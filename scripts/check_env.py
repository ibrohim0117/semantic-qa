"""Muhitning yakuniy tekshiruvi: hamma qism birga ishlaydimi.

3 ta kartochkadan iborat kichik tizim quriladi va butun yo'l bosib o'tiladi:
savol -> bge-m3 -> Qdrant -> answer_id -> Postgres -> javob matni.
Sinov ma'lumotlari vaqtinchalik: TEMP jadval va oxirida o'chiriladigan kolleksiya.

Ishga tushirish: poetry run python scripts/check_env.py
"""

import asyncio
import os
import resource
import subprocess
import time

import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+asyncpg://semantic:semantic@localhost:5432/semantic_qa"
)
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
MODEL_PATH = "models/bge-m3-fp16"  # scripts/convert_model_fp16.py yaratadi
COLLECTION = "env_check"  # vaqtinchalik sinov kolleksiyasi

# Kichik "kartochkalar": (javob_id, javob matni, savol)
CARDS = [
    (
        1,
        "Ariza my.gov.uz yoki Davlat xizmatlari markazi orqali beriladi.",
        "Pasportni (ID-karta) almashtirish tartibi qanday?",
    ),
    (
        2,
        "Ariza, davlat boji kvitansiyasi va eski hujjat kerak.",
        "ID-karta olish uchun qanday hujjatlar kerak?",
    ),
    (
        3,
        "Odatda 1 ish kuni ichida tayyor bo'ladi.",
        "ID-karta necha kunda tayyor bo'ladi?",
    ),
]
NEW_QUESTION = "pasportimni yangisiga qanday almashtiraman?"


def ok(msg: str) -> None:
    print(f"   ✅ {msg}")


async def main() -> None:
    # 1. Postgres: ulanish va vaqtinchalik jadvalga javoblarni yozish
    print("1) PostgreSQL (SQLAlchemy + asyncpg)")
    engine = create_async_engine(DATABASE_URL)
    async with engine.connect() as conn:
        version = (await conn.execute(text("SHOW server_version"))).scalar()
        ok(f"ulandi, versiya {version}")
        await conn.execute(text("CREATE TEMP TABLE answers_check (id int PRIMARY KEY, body text)"))
        await conn.execute(
            text("INSERT INTO answers_check VALUES (:id, :body)"),
            [{"id": a_id, "body": body} for a_id, body, _ in CARDS],
        )
        count = (await conn.execute(text("SELECT count(*) FROM answers_check"))).scalar()
        ok(f"{count} ta javob vaqtinchalik jadvalga yozildi")

        # 2. bge-m3: savollarni vektorga aylantirish
        print("\n2) bge-m3 (sentence-transformers + PyTorch)")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        # GPU'da fp16: xotira 2 barobar kam, baholar deyarli bir xil (o'lchangan farq 0.0004).
        # CPU'da fp16 sekin, shuning uchun u yerda fp32 qoladi.
        dtype = torch.float16 if device != "cpu" else torch.float32
        t0 = time.perf_counter()
        model = SentenceTransformer(MODEL_PATH, device=device, model_kwargs={"dtype": dtype})
        ok(f"model yuklandi: {time.perf_counter() - t0:.1f} s, qurilma: {device}, aniqlik: {dtype}")
        vectors = model.encode([q for _, _, q in CARDS], normalize_embeddings=True)
        ok(f"{len(vectors)} ta savol vektorga aylandi, har biri {vectors.shape[1]} ta son")

        # 3. Qdrant: kolleksiya yaratish va vektorlarni yozish
        print("\n3) Qdrant (qdrant-client)")
        qdrant = QdrantClient(url=QDRANT_URL)
        if qdrant.collection_exists(COLLECTION):
            qdrant.delete_collection(COLLECTION)
        qdrant.create_collection(
            COLLECTION, vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )
        qdrant.upsert(
            COLLECTION,
            points=[
                PointStruct(
                    id=i,
                    vector=vec.tolist(),
                    payload={"answer_id": a_id, "question": q},
                )
                for i, ((a_id, _, q), vec) in enumerate(zip(CARDS, vectors, strict=True), start=1)
            ],
        )
        ok(f"kolleksiya yaratildi, {qdrant.count(COLLECTION).count} ta nuqta yozildi")

        # 4. Butun yo'l: yangi savol -> vektor -> qidiruv -> javob
        print("\n4) Butun yo'l")
        print(f"   Yangi savol: «{NEW_QUESTION}»")
        q_vec = model.encode(NEW_QUESTION, normalize_embeddings=True)
        hits = qdrant.query_points(COLLECTION, query=q_vec.tolist(), limit=3).points
        for h in hits:
            print(f"   {h.score:.3f}  {h.payload['question']}  (javob {h.payload['answer_id']})")
        best = hits[0].payload["answer_id"]
        body = (
            await conn.execute(text("SELECT body FROM answers_check WHERE id = :id"), {"id": best})
        ).scalar()
        ok(f"g'olib: {best}-javob → Postgres'dan olingan matn: «{body}»")

        qdrant.delete_collection(COLLECTION)
        ok("sinov kolleksiyasi o'chirildi (vaqtinchalik jadval ulanish bilan o'zi o'chadi)")
    await engine.dispose()


def run(cmd: list[str]) -> str:
    """Tizim buyrug'ini bajarib, chiqishini qaytaradi (xato bo'lsa ham to'xtamaydi)."""
    return subprocess.run(cmd, check=False, capture_output=True, text=True).stdout


def report_resources() -> None:
    """5. Resurslar: hamma narsa birga ishlaganda."""
    print("\n5) Resurslar (model + Postgres + Qdrant birga)")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3  # macOS: bayt
    gpu = torch.mps.current_allocated_memory() / 1024**3
    print(f"   Python jarayoni: {rss:.2f} GB, GPU (MPS): {gpu:.2f} GB")
    pressure = run(["memory_pressure"]).splitlines()
    free_line = next((ln for ln in pressure if "free percentage" in ln), "?")
    print(f"   Mac: {free_line.strip()}")
    print(f"   Swap: {run(['sysctl', '-n', 'vm.swapusage']).strip()}")
    stats = run(["docker", "stats", "--no-stream", "--format", "{{.Name}}: {{.MemUsage}}"])
    for line in stats.strip().splitlines():
        if line.startswith("semantic-qa"):
            print(f"   Docker {line}")


if __name__ == "__main__":
    asyncio.run(main())
    report_resources()
