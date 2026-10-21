"""1.1-qadam: ma'lumotni ko'zdan kechirish.

data/raw dagi CSV'lar o'qiladi va o'zgartirilmaydi. Natija faqat ekranga chiqadi.
Ishga tushirish: poetry run python scripts/explore_data.py
"""

import re

import pandas as pd

ANSWERS_CSV = "data/raw/answers.csv"
QUESTIONS_CSV = "data/raw/questions.csv"

# Apostrof turlari: o'zbek tilida o' va g' harflari shular bilan yoziladi
APOSTROPHES = {
    "'": "oddiy (ASCII)",
    "‘": "chap qo'shtirnoq",
    "’": "o'ng qo'shtirnoq",
    "ʻ": "o'zbek okina",
    "ʼ": "modifikator",
    "`": "teskari",
}
CYRILLIC = re.compile(r"[А-Яа-яЁёЎўҚқҒғҲҳ]")


def title(text: str) -> None:
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


def structure(answers: pd.DataFrame, questions: pd.DataFrame) -> None:
    title("1. TUZILISH")
    for name, df in [("answers.csv", answers), ("questions.csv", questions)]:
        print(f"\n{name}: {len(df)} qator, ustunlar: {list(df.columns)}")
        print(df.head(3).to_string(max_colwidth=60))

    missing = set(questions["answer_id"]) - set(answers["id"])
    print("\nBog'lanish: questions.answer_id → answers.id")
    print(f"Mavjud bo'lmagan javobga ishora qiluvchi savollar: {len(missing)}")


def distribution(answers: pd.DataFrame, questions: pd.DataFrame) -> None:
    title("2. TAQSIMOT")
    per_answer = questions.groupby("answer_id").size()
    print("\nJavobga nechta savol to'g'ri keladi (savollar soni → nechta javobda):")
    print(per_answer.value_counts().sort_index().to_string())

    n_cat, n_topic = answers["category_id"].nunique(), answers["topic_id"].nunique()
    print(f"\nKategoriyalar: {n_cat} ta, mavzular: {n_topic} ta")
    print("Har kategoriyada nechta javob:")
    print(answers["category_id"].value_counts().sort_index().to_string())

    topic_cats = answers.groupby("topic_id")["category_id"].nunique()
    print(f"\nHar bir mavzu faqat bitta kategoriyaga tegishlimi: {bool((topic_cats == 1).all())}")

    print(f"\nSanalar (created_at): {sorted(answers['created_at'].unique())}")
    print(
        f"is_active qiymatlari: {questions['is_active'].unique().tolist()}, "
        f"status qiymatlari: {answers['status'].unique().tolist()}"
    )


def lengths(answers: pd.DataFrame, questions: pd.DataFrame) -> None:
    title("3. UZUNLIK")
    for name, series in [("Savol", questions["question"]), ("Javob", answers["answer"])]:
        chars = series.str.len()
        words = series.str.split().str.len()
        print(f"\n{name}:")
        print(f"  belgilar min {chars.min()} / median {chars.median():.0f} / max {chars.max()}")
        print(f"  so'zlar  min {words.min()} / median {words.median():.0f} / max {words.max()}")
    longest = answers.loc[answers["answer"].str.len().idxmax()]
    print(f"\nEng uzun javob (id {longest['id']}, {len(longest['answer'])} belgi):")
    print(f"  {longest['answer'][:200]}…")


def quality(answers: pd.DataFrame, questions: pd.DataFrame) -> None:
    title("4. SIFAT")
    print(
        f"\nBo'sh qiymatlar: answers {int(answers.isna().sum().sum())}, "
        f"questions {int(questions.isna().sum().sum())}"
    )

    q_norm = questions["question"].str.lower().str.replace(r"[^\w\s]", "", regex=True).str.strip()
    print(f"Aynan takror savollar: {questions['question'].duplicated().sum()}")
    print(f"Tinish belgisi va harf kattaligisiz takror: {q_norm.duplicated().sum()}")
    dup = questions[q_norm.duplicated(keep=False)].assign(norm=q_norm)
    cross = dup.groupby("norm")["answer_id"].nunique()
    print(f"  shulardan TURLI javoblarga bog'langan (chalkashlik xavfi): {(cross > 1).sum()}")
    print(f"Aynan takror javoblar: {answers['answer'].duplicated().sum()}")

    print("\nApostrof turlari (necha marta uchraydi):")
    for ch, label in APOSTROPHES.items():
        q = questions["question"].str.count(re.escape(ch)).sum()
        a = answers["answer"].str.count(re.escape(ch)).sum()
        if q or a:
            print(f"  {ch!r:6} {label:<18} savollarda {q:>5}, javoblarda {a:>5}")

    q_cyr = questions["question"].str.contains(CYRILLIC).sum()
    a_cyr = answers["answer"].str.contains(CYRILLIC).sum()
    print(f"\nKirill harfli matnlar: savollarda {q_cyr}, javoblarda {a_cyr}")

    # Shablonlar: savollarning eng ko'p takrorlanadigan oxirgi 3 so'zi
    endings = questions["question"].str.split().str[-3:].str.join(" ")
    print("\nEng ko'p uchraydigan savol oxirlari (shablon belgisi):")
    print(endings.value_counts().head(10).to_string())

    print("\nMisol: bitta javobning barcha savol variantlari (answer_id = 3):")
    for q in questions.loc[questions["answer_id"] == 3, "question"]:
        print(f"  - {q}")


TEMPLATE = (
    r"(?:haqida ma'lumot bering|bo'yicha ma'lumot kerak|^Menga .* haqida ayting|^Iltimos, ayting:)"
)


def generation(answers: pd.DataFrame, questions: pd.DataFrame) -> None:
    title("5. SAVOLLAR QANDAY YASALGAN")
    first = questions.sort_values("id").groupby("answer_id").head(1)
    print(
        f"\nHar javobning BIRINCHI savoli shablon iborali: "
        f"{first['question'].str.contains(TEMPLATE).sum()} / {len(first)}"
    )
    print(
        f"Barcha savollardan shablon iborali: "
        f"{questions['question'].str.contains(TEMPLATE).sum()} / {len(questions)}"
    )

    later_q = questions["created_at"].str.startswith("2026-08-11")
    later_a = answers["created_at"].str.startswith("2026-08-11")
    target_ids = sorted(questions.loc[later_q, "answer_id"].unique().tolist())
    print(f"\n11-avgustda qo'shilganlar: javoblar {later_a.sum()}, savollar {later_q.sum()}")
    print(f"  bu savollar qaysi javoblarga tegishli: {target_ids}")

    per_answer = questions.groupby("answer_id").size()
    print(f"12 savolli javoblar: {per_answer[per_answer == 12].index.tolist()}")


def main() -> None:
    answers = pd.read_csv(ANSWERS_CSV)
    questions = pd.read_csv(QUESTIONS_CSV)
    structure(answers, questions)
    distribution(answers, questions)
    lengths(answers, questions)
    quality(answers, questions)
    generation(answers, questions)


if __name__ == "__main__":
    main()
