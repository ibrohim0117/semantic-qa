"""1.3-qadam: test savollari bazadagi savollarning "ko'chirmasi" emasligini tekshirish.

Jaccard o'xshashligi = umumiy so'zlar / barcha turli so'zlar (0 = umumiy so'z yo'q, 1 = bir xil).
Ma'no emas, aynan so'zlar solishtiriladi. Chegaradan yuqori savollar qayta yoziladi.
Taqqoslash uchun bazaning o'zida asl savol va uning shablon egizaklari o'lchanadi.

Ishga tushirish: poetry run python scripts/check_test_overlap.py eval/test_b.csv
"""

import re
import sys

import pandas as pd

THRESHOLD = 0.5
APOSTROPHES = re.compile(r"[‘’ʻʼ`´]")


def words(text: str) -> set[str]:
    text = APOSTROPHES.sub("'", text.lower())
    return set(re.findall(r"[\w']+", text))


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def main() -> None:
    test_path = sys.argv[1] if len(sys.argv) > 1 else "eval/test_b.csv"
    test = pd.read_csv(test_path)
    questions = pd.read_csv("data/raw/questions.csv")
    base = [words(q) for q in questions["question"]]

    # 1. Test savollari: har biri uchun bazadagi eng o'xshash savol
    rows = []
    for q in test["question"]:
        scores = [jaccard(words(q), b) for b in base]
        best = max(range(len(base)), key=scores.__getitem__)
        rows.append((scores[best], q, questions["question"].iloc[best]))
    max_scores = pd.Series([r[0] for r in rows])

    # 2. Taqqoslash: bazadagi asl savol va uning shablon egizaklari
    original = questions.sort_values("id").groupby("answer_id").head(1)
    sibling_scores = []
    for _, row in original.iterrows():
        siblings = questions[
            (questions["answer_id"] == row["answer_id"]) & (questions["id"] != row["id"])
        ]
        sibling_scores.append(
            max(jaccard(words(row["question"]), words(s)) for s in siblings["question"])
        )
    siblings = pd.Series(sibling_scores)

    print(f"Test fayli: {test_path} ({len(test)} ta savol)\n")
    print("Bazadagi eng o'xshash savol bilan so'z o'xshashligi (Jaccard):")
    for label, series in [("Test savollari", max_scores), ("Asl ↔ egizaklari", siblings)]:
        print(f"  {label:<17} median {series.median():.2f}, max {series.max():.2f}")
    flagged = [r for r in rows if r[0] > THRESHOLD]
    print(f"\nChegaradan ({THRESHOLD}) yuqori test savollari: {len(flagged)} ta")
    for score, q, nearest in flagged:
        print(f"  {score:.2f}  «{q}»\n        bazada: «{nearest}»")


if __name__ == "__main__":
    main()
