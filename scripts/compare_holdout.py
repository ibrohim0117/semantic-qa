"""1.2-qadam tajribasi: qaysi savolni yashirish testni qanchalik o'zgartiradi.

Har javobdan bitta savol yashiriladi (test), qolganlari bazada qoladi (indeks).
Ikki yo'l solishtiriladi: tasodifiy savol va asl (birinchi) savol.
Vektorlar scripts/try_question.py yaratgan keshdan olinadi.

Ishga tushirish: poetry run python scripts/compare_holdout.py
"""

import numpy as np
import pandas as pd

CACHE = "data/processed/demo_question_vectors.npy"  # questions.csv tartibida
SEED = 42


def evaluate(
    questions: pd.DataFrame,
    vectors: np.ndarray,
    test_mask: np.ndarray,
    index_mask: np.ndarray | None = None,
) -> dict:
    """Test savollarini indeksdan qidirib, recall@1 va recall@3 ni hisoblaydi.

    index_mask berilmasa, testga kirmagan hamma savol indeksga kiradi.
    """
    if index_mask is None:
        index_mask = ~test_mask
    index_vecs = vectors[index_mask]
    index_answer_ids = questions.loc[index_mask, "answer_id"].to_numpy()
    test = questions[test_mask]

    hits1, hits3, misses = 0, 0, []
    for vec, (_, row) in zip(vectors[test_mask], test.iterrows(), strict=True):
        scores = index_vecs @ vec
        # Javob bo'yicha guruhlash: har javobga uning savollari ichidagi eng yuqori ball
        best = pd.Series(scores).groupby(index_answer_ids).max().sort_values(ascending=False)
        top3 = best.index[:3].tolist()
        hits1 += top3[0] == row["answer_id"]
        hits3 += row["answer_id"] in top3
        if top3[0] != row["answer_id"]:
            misses.append((row["question"], row["answer_id"], top3[0]))
    n = len(test)
    return {"n": n, "recall@1": hits1 / n, "recall@3": hits3 / n, "misses": misses}


def main() -> None:
    questions = pd.read_csv("data/raw/questions.csv")
    vectors = np.load(CACHE)

    # 1-yo'l: har javobdan tasodifiy bitta savol
    random_ids = questions.groupby("answer_id").sample(n=1, random_state=SEED)["id"]
    # 2-yo'l: har javobdan asl savol (id bo'yicha birinchisi)
    original_ids = questions.sort_values("id").groupby("answer_id").head(1)["id"]

    # 3-yo'l: asl savol yashiriladi VA uning so'zma-so'z nusxasi ("Iltimos, ayting: <asl>")
    # ham indeksdan olinadi
    copies = questions["question"].str.startswith("Iltimos, ayting:").to_numpy()
    original_mask = questions["id"].isin(original_ids).to_numpy()

    scenarios = [
        ("1) Tasodifiy savol yashirildi", questions["id"].isin(random_ids).to_numpy(), None),
        ("2) Asl savol yashirildi", original_mask, None),
        ("3) Asl savol + uning nusxasi olib tashlandi", original_mask, ~original_mask & ~copies),
    ]
    for name, test_mask, index_mask in scenarios:
        result = evaluate(questions, vectors, test_mask, index_mask)
        print(f"\n{name}: {result['n']} ta test savoli")
        print(f"  recall@1 = {result['recall@1']:.1%}   recall@3 = {result['recall@3']:.1%}")
        print(f"  Xato javoblar: {len(result['misses'])} ta. Misollar:")
        for question, true_id, got_id in result["misses"][:3]:
            print(f"    «{question}»  kutilgan: {true_id}, tizim berdi: {got_id}")


if __name__ == "__main__":
    main()
