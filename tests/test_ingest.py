"""src/ingest.py testlari: validatsiya va turlarni o'girish.

Bazasiz ishlaydi: har test kichik soxta ma'lumot (2 javob, 3 savol) yasaydi,
uni ataylab buzadi va validate() buni to'g'ri aniqlashini tekshiradi.
"""

import pandas as pd

from src.ingest import QUESTION_COLS, to_rows, validate

TS = pd.Timestamp("2026-08-04")


def make_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    answers = pd.DataFrame(
        {
            "id": [1, 2],
            "category_id": [1, 1],
            "topic_id": [1, 1],
            "text": ["Birinchi javob", "Ikkinchi javob"],
            "status": ["active", "active"],
            "created_at": [TS, TS],
            "updated_at": [TS, TS],
        }
    )
    questions = pd.DataFrame(
        {
            "id": [10, 11, 12],
            "answer_id": [1, 1, 2],
            "text": ["Savol a", "Savol b", "Savol c"],
            "is_active": [True, True, True],
            "created_at": [TS, TS, TS],
            "updated_at": [TS, TS, TS],
        }
    )
    return answers, questions


def test_valid_data_passes():
    report = validate(*make_data())
    assert report.errors == []
    assert report.warnings == []


def test_orphan_question_is_error():
    answers, questions = make_data()
    questions.loc[len(questions)] = [13, 99, "Egasiz savol", True, TS, TS]
    report = validate(answers, questions)
    assert any("[99]" in e for e in report.errors)


def test_duplicate_id_is_error():
    answers, questions = make_data()
    questions.loc[len(questions)] = [10, 1, "Takroriy id", True, TS, TS]
    report = validate(answers, questions)
    assert any("takroriy id" in e for e in report.errors)


def test_empty_text_is_error():
    answers, questions = make_data()
    answers.loc[0, "text"] = "   "
    report = validate(answers, questions)
    assert any("bo'sh matn" in e for e in report.errors)


def test_missing_value_is_error():
    answers, questions = make_data()
    questions.loc[0, "text"] = None
    report = validate(answers, questions)
    assert any("questions.text" in e for e in report.errors)


def test_missing_column_is_error():
    answers, questions = make_data()
    report = validate(answers.drop(columns=["status"]), questions)
    assert any("status" in e for e in report.errors)


def test_answer_without_questions_is_only_warning():
    answers, questions = make_data()
    answers.loc[len(answers)] = [3, 1, 1, "Savolsiz javob", "active", TS, TS]
    report = validate(answers, questions)
    assert report.errors == []
    assert any("[3]" in w for w in report.warnings)


def test_mixed_script_word_is_only_warning():
    answers, questions = make_data()
    answers.loc[0, "text"] = "Yangi egа murojaat qiladi"  # "egа" oxiridagi "а" kirillcha
    report = validate(answers, questions)
    assert report.errors == []
    assert any("aralash" in w for w in report.warnings)


def test_to_rows_gives_plain_python_types():
    # asyncpg numpy turlarini qabul qilmaydi, oddiy int/bool kerak
    _, questions = make_data()
    row = to_rows(questions, QUESTION_COLS)[0]
    assert type(row["id"]) is int
    assert type(row["is_active"]) is bool
