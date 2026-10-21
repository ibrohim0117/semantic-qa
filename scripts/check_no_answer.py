"""1.4-qadam: "bazada javobi yo'q" nomzod savollarni tekshirish.

Har savol uchun tizim eng yaqin deb topadigan 3 ta javob chiqariladi.
Ular savolga haqiqatan javob bo'ladimi, odam o'qib hal qiladi.
Vektorlar scripts/try_question.py yaratgan keshdan olinadi.

Ishga tushirish:
    poetry run python scripts/check_no_answer.py data/processed/c_candidates.csv
"""

import sys

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

MODEL_PATH = "models/bge-m3-fp16"
CACHE = "data/processed/demo_question_vectors.npy"


def main() -> None:
    candidates = pd.read_csv(sys.argv[1])
    answers = pd.read_csv("data/raw/answers.csv").set_index("id")
    questions = pd.read_csv("data/raw/questions.csv")
    vectors = np.load(CACHE)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = SentenceTransformer(MODEL_PATH, device=device, model_kwargs={"dtype": dtype})
    q_vecs = model.encode(candidates["question"].tolist(), normalize_embeddings=True)

    for i, (vec, row) in enumerate(zip(q_vecs, candidates.itertuples(), strict=True), start=1):
        best = (
            pd.Series(vectors @ vec)
            .groupby(questions["answer_id"].to_numpy())
            .max()
            .sort_values(ascending=False)
            .head(3)
        )
        print(f"\n{i:>2}. [{row.type}] {row.question}")
        for answer_id, score in best.items():
            print(
                f"     {score:.3f}  javob {answer_id:>3}: {answers.loc[answer_id, 'answer'][:150]}"
            )


if __name__ == "__main__":
    main()
