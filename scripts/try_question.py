"""Istalgan savolni butun bazada sinab ko'rish (Qdrant'siz, oddiy numpy bilan).

Bazadagi 3048 savol vektorga aylantiriladi (birinchi marta ~1 daqiqa, keyin keshdan),
foydalanuvchi savoli ular bilan solishtiriladi va eng yaqinlari ko'rsatiladi.

Ishga tushirish:
    poetry run python scripts/try_question.py "hozir menda yashil pasport bor, qachon yangilashim kerak?"
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

MODEL_PATH = "models/bge-m3-fp16"
CACHE = Path("data/processed/demo_question_vectors.npy")
TOP_K = 10


def load_model() -> SentenceTransformer:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device != "cpu" else torch.float32
    return SentenceTransformer(MODEL_PATH, device=device, model_kwargs={"dtype": dtype})


def main() -> None:
    query = " ".join(sys.argv[1:]) or "hozir menda yashil pasport bor, qachon yangilashim kerak?"
    answers = pd.read_csv("data/raw/answers.csv").set_index("id")
    questions = pd.read_csv("data/raw/questions.csv")
    model = load_model()

    # 1-bosqich (tayyorlash): bazadagi barcha savollar -> vektorlar
    if CACHE.exists():
        vectors = np.load(CACHE)
    else:
        print("Bazadagi 3048 savol vektorga aylantirilmoqda (bir marta)...")
        vectors = model.encode(
            questions["question"].tolist(), normalize_embeddings=True, batch_size=64
        )
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.save(CACHE, vectors)

    # 2-bosqich (savol kelganda): savol -> vektor -> hamma savol bilan solishtirish
    q_vec = model.encode(query, normalize_embeddings=True)
    scores = vectors @ q_vec  # 3048 ta baho, har bir saqlangan savol uchun bittadan
    top = np.argsort(-scores)[:TOP_K]

    print(f"\nFoydalanuvchi savoli: «{query}»")
    print(f"\nBazadagi {len(questions)} ta savol ichidan eng yaqin {TOP_K} tasi:")
    for rank, i in enumerate(top, start=1):
        row = questions.iloc[i]
        print(f"  {rank:>2}. {scores[i]:.3f}  [javob {row['answer_id']:>3}]  {row['question']}")

    # Javob bo'yicha guruhlash: har javobga uning savollari ichidagi eng yuqori ball
    best_per_answer = (
        pd.DataFrame({"answer_id": questions["answer_id"], "score": scores})
        .groupby("answer_id")["score"]
        .max()
        .sort_values(ascending=False)
        .head(3)
    )
    print("\nJavoblar bo'yicha guruhlangandan keyin (eng yaxshi 3 ta):")
    for answer_id, score in best_per_answer.items():
        print(f"  {score:.3f}  javob {answer_id}: {answers.loc[answer_id, 'answer'][:90]}…")

    winner = best_per_answer.index[0]
    print(f"\nFoydalanuvchiga qaytariladigan javob ({winner}):\n  {answers.loc[winner, 'answer']}")


if __name__ == "__main__":
    main()
