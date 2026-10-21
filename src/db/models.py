"""PostgreSQL jadvallari: SQLAlchemy klasslari ko'rinishida.

Har bir klass = bitta jadval, har bir atribut = bitta ustun.
Vektorlar bu yerda saqlanmaydi (ular Qdrant'da), faqat matnlar va bog'lanishlar.
"""

from datetime import datetime

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Answer(Base):
    __tablename__ = "answers"

    # CSV'dagi id saqlanadi: test fayllari va Qdrant shu raqamlarga tayanadi
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=False)
    category_id: Mapped[int]
    topic_id: Mapped[int]
    text: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]

    questions: Mapped[list["Question"]] = relationship(back_populates="answer")


class Question(Base):
    __tablename__ = "questions"

    # Qdrant'dagi nuqta id'si ham shu raqam bo'ladi
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=False)
    # Javob o'chirilsa, uning savollari ham o'chadi (egasiz savol qolmaydi)
    answer_id: Mapped[int] = mapped_column(ForeignKey("answers.id", ondelete="CASCADE"), index=True)
    text: Mapped[str] = mapped_column(Text)
    is_active: Mapped[bool]
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]

    answer: Mapped[Answer] = relationship(back_populates="questions")
