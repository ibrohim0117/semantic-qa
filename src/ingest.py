"""CSV'dagi javob va savollarni PostgreSQL'ga yuklash.

Bosqichlar: o'qish -> tekshirish (validatsiya) -> bitta tranzaksiyada yozish.
- Xato topilsa, bazaga hech narsa yozilmaydi (chala ma'lumot xato xabaridan xavfliroq).
- Idempotent: necha marta ishga tushirilsa ham baza CSV bilan bir xil bo'lib qoladi
  (yangi qator qo'shiladi, bori yangilanadi, CSV'dan olib tashlangani o'chiriladi).

Ishga tushirish: poetry run python -m src.ingest [data/raw]
"""

import asyncio
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Answer, Question
from src.db.session import SessionLocal, engine

DATA_DIR = Path("data/raw")
ANSWER_COLS = ["id", "category_id", "topic_id", "text", "status", "created_at", "updated_at"]
QUESTION_COLS = ["id", "answer_id", "text", "is_active", "created_at", "updated_at"]
# Bitta INSERT'dagi qatorlar soni. asyncpg bitta so'rovda ~32 ming parametrgacha qabul qiladi.
CHUNK = 1000
# Bitta so'z ichida lotin va kirill harflari aralashgan (masalan "egа" — oxirgi "а" kirillcha)
MIXED_SCRIPT = re.compile(r"\b(?=\w*[A-Za-z])(?=\w*[А-Яа-яЁёЎўҚқҒғҲҳ])\w+\b")


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)  # yuklashni to'xtatadi
    warnings: list[str] = field(default_factory=list)  # faqat xabar beriladi


def load_csv(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = ["created_at", "updated_at"]
    answers = pd.read_csv(data_dir / "answers.csv", parse_dates=dates)
    questions = pd.read_csv(data_dir / "questions.csv", parse_dates=dates)
    return answers.rename(columns={"answer": "text"}), questions.rename(
        columns={"question": "text"}
    )


def validate(answers: pd.DataFrame, questions: pd.DataFrame) -> Report:
    report = Report()
    for name, df, cols in [
        ("answers", answers, ANSWER_COLS),
        ("questions", questions, QUESTION_COLS),
    ]:
        missing = set(cols) - set(df.columns)
        if missing:
            report.errors.append(f"{name}: ustun(lar) yo'q: {sorted(missing)}")
            continue
        for col, n in df[cols].isna().sum().items():
            if n:
                report.errors.append(f"{name}.{col}: {n} ta bo'sh qiymat")
        if (dup := df["id"].duplicated().sum()) > 0:
            report.errors.append(f"{name}: {dup} ta takroriy id")
        if (empty := (df["text"].astype(str).str.strip() == "").sum()) > 0:
            report.errors.append(f"{name}: {empty} ta bo'sh matn")
    if report.errors:
        return report

    orphans = sorted(set(questions["answer_id"]) - set(answers["id"]))
    if orphans:
        report.errors.append(
            f"{len(orphans)} xil mavjud bo'lmagan javobga savol bog'langan: {orphans[:10]}"
        )
    no_questions = sorted(set(answers["id"]) - set(questions["answer_id"]))
    if no_questions:
        report.warnings.append(f"savoli yo'q javoblar (qidiruvda topilmaydi): {no_questions[:10]}")
    for name, df in [("answers", answers), ("questions", questions)]:
        mixed = df.loc[df["text"].str.contains(MIXED_SCRIPT), "id"].tolist()
        if mixed:
            report.warnings.append(f"{name}: lotin va kirill aralash so'z bor, id: {mixed}")
    return report


def to_rows(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    # astype(object): numpy sonlarini oddiy Python int/bool ga o'giradi (asyncpg shuni kutadi)
    return df[cols].astype(object).to_dict("records")


async def upsert(session: AsyncSession, model: type, rows: list[dict], cols: list[str]) -> None:
    """INSERT ... ON CONFLICT (id) DO UPDATE: yangi qator qo'shiladi, bori yangilanadi."""
    for start in range(0, len(rows), CHUNK):
        stmt = insert(model).values(rows[start : start + CHUNK])
        stmt = stmt.on_conflict_do_update(
            index_elements=[model.id],
            set_={col: stmt.excluded[col] for col in cols if col != "id"},
        )
        await session.execute(stmt)


async def sync(answers: pd.DataFrame, questions: pd.DataFrame) -> dict[str, int]:
    answer_ids = [int(i) for i in answers["id"]]
    question_ids = [int(i) for i in questions["id"]]

    # session.begin(): blok xatosiz tugasa COMMIT, xato bo'lsa ROLLBACK (hammasi yoki hech narsa)
    async with SessionLocal() as session, session.begin():
        # Tartib muhim: savollar javoblarga ishora qiladi, shuning uchun avval javoblar
        await upsert(session, Answer, to_rows(answers, ANSWER_COLS), ANSWER_COLS)
        await upsert(session, Question, to_rows(questions, QUESTION_COLS), QUESTION_COLS)

        # CSV'dan olib tashlangan qatorlar bazadan ham o'chiriladi
        deleted_q = await session.execute(delete(Question).where(Question.id.not_in(question_ids)))
        deleted_a = await session.execute(delete(Answer).where(Answer.id.not_in(answer_ids)))

        return {
            "answers": await session.scalar(select(func.count()).select_from(Answer)),
            "questions": await session.scalar(select(func.count()).select_from(Question)),
            "deleted_answers": deleted_a.rowcount,
            "deleted_questions": deleted_q.rowcount,
        }


async def main(data_dir: Path) -> int:
    answers, questions = load_csv(data_dir)
    print(f"O'qildi: {len(answers)} javob, {len(questions)} savol ({data_dir})")

    report = validate(answers, questions)
    for warning in report.warnings:
        print(f"  ⚠️  {warning}")
    if report.errors:
        for error in report.errors:
            print(f"  ❌ {error}")
        print("Xato topildi, bazaga hech narsa yozilmadi.")
        return 1

    result = await sync(answers, questions)
    await engine.dispose()
    print(f"Bazada: {result['answers']} javob, {result['questions']} savol")
    print(
        f"CSV'da yo'qligi uchun o'chirildi: {result['deleted_answers']} javob, "
        f"{result['deleted_questions']} savol"
    )
    return 0


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_DIR
    sys.exit(asyncio.run(main(path)))
