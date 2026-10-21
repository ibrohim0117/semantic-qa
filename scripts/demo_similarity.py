"""bge-m3 qanday ishlashini ko'rsatuvchi demo.

Gap -> tokenlar -> 1024 sonli vektor -> cosine o'xshashlik bahosi.
Ishga tushirish: poetry run python scripts/demo_similarity.py
QUERY va SENTENCES ni o'zgartirib, o'zingiz sinab ko'ring.
"""

import resource
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_PATH = "models/bge-m3-fp16"  # scripts/convert_model_fp16.py yaratadi

# Foydalanuvchi savoli
QUERY = "pasportimni qanday almashtiraman?"

# Solishtiriladigan gaplar: (gap, izoh)
SENTENCES = [
    ("Pasportni (ID-karta) almashtirish tartibi qanday?", "ma'nodosh, bazadan"),
    ("Паспортни қандай алмаштираман?", "ma'nodosh, kirillda"),
    ("Как заменить паспорт?", "ma'nodosh, ruscha"),
    ("pasport almashtrsh qanaqa", "ma'nodosh, imlo xatolari bilan"),
    ("ID-karta olish uchun qanday hujjatlar kerak?", "mavzu yaqin, savol boshqa"),
    ("ID-karta necha kunda tayyor bo'ladi?", "mavzu yaqin, savol boshqa"),
    ("Ertaga yomg'ir yog'adimi?", "umuman aloqasiz"),
]


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Modelni diskdan xotiraga yuklash (internet kerak emas)
    t0 = time.perf_counter()
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = SentenceTransformer(MODEL_PATH, device=device, model_kwargs={"dtype": dtype})
    print(f"Model yuklandi: {time.perf_counter() - t0:.1f} s, qurilma: {device}\n")

    # 1-bosqich: tokenizatsiya
    tokenizer = model.tokenizer
    ids = tokenizer(QUERY)["input_ids"]
    print("1) TOKENLAR")
    print(f"   Gap:       {QUERY}")
    print(f"   Tokenlar:  {tokenizer.convert_ids_to_tokens(ids)}")
    print(f"   Raqamlari: {ids}")
    print(f"   Lug'at hajmi: {tokenizer.vocab_size:,} token\n")

    # 2-3-bosqich: vektor va normalizatsiya
    q_vec = model.encode(QUERY, normalize_embeddings=True)
    print("2) VEKTOR")
    print(f"   O'lchami: {q_vec.shape[0]} ta son")
    print(f"   Birinchi 8 tasi: {[round(float(x), 3) for x in q_vec[:8]]}")
    print(f"   Uzunligi (normallashtirilgan): {np.linalg.norm(q_vec):.4f}\n")

    # 4-bosqich: cosine o'xshashlik
    texts = [s for s, _ in SENTENCES]
    t0 = time.perf_counter()
    vecs = model.encode(texts, normalize_embeddings=True)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    # Normallashtirilgan vektorlarda cosine = mos sonlar ko'paytmalarining yig'indisi
    scores = vecs @ q_vec

    print("3) O'XSHASHLIK BAHOLARI (yuqoridan pastga)")
    print(f"   Savol: «{QUERY}»\n")
    for i in np.argsort(-scores):
        text, note = SENTENCES[i]
        bar = "█" * int(scores[i] * 30)
        print(f"   {scores[i]:.3f} {bar:<30} {text}  ({note})")

    # Qo'lda hisoblab tekshirish: 1024 ta ko'paytmaning yig'indisi
    manual = sum(float(a) * float(b) for a, b in zip(q_vec, vecs[0], strict=True))
    print(f"\n   Qo'lda tekshiruv (1-gap): 1024 ko'paytma yig'indisi = {manual:.3f}")

    # Resurslar
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3  # macOS: bayt
    print("\n4) RESURSLAR")
    print(f"   {len(texts)} ta gapni vektorlash: {elapsed_ms:.0f} ms")
    print(f"   Jarayon egallagan eng ko'p xotira: {rss_gb:.2f} GB")
    if device == "mps":
        print(f"   GPU xotirasi (MPS): {torch.mps.current_allocated_memory() / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
