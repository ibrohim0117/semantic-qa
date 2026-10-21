"""1.3-qadam: B test to'plami uchun javoblarni tanlash.

Har kategoriyadan o'z ulushiga qarab (stratifikatsiya) ~80 ta javob tanlanadi.
Savollar YOZILMAYDI va CHIQARILMAYDI: test savollari faqat javob matnidan yoziladi,
shunda ular bazadagi shablon savollarga o'xshab qolmaydi.

Ishga tushirish: poetry run python scripts/sample_answers_for_eval.py
Natija: data/processed/b_answers.csv
"""

from pathlib import Path

import pandas as pd

SEED = 42
FRACTION = 80 / 505
# Bu javoblarning savollari ishlab chiqish jarayonida ko'rib chiqilgan.
# Ular uchun yozilgan test savoli beixtiyor o'sha savollarga o'xshab qolishi mumkin.
SEEN_ANSWER_IDS = {1, 2, 3, 6, 22, 27, 34, 40, 56, 85, 144, 182, 328, 338, 358, 383, 384, 505}
OUTPUT = Path("data/processed/b_answers.csv")


def main() -> None:
    answers = pd.read_csv("data/raw/answers.csv")
    pool = answers[~answers["id"].isin(SEEN_ANSWER_IDS)]
    sample = pool.groupby("category_id").sample(frac=FRACTION, random_state=SEED)
    sample = sample.sort_values("id")[["id", "category_id", "answer"]]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUTPUT, index=False)
    print(f"Tanlandi: {len(sample)} ta javob → {OUTPUT}")
    print(sample["category_id"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
