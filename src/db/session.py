"""Bazaga ulanish: bitta engine va undan sessiyalar yaratuvchi fabrika.

Ikkalasi ham birinchi kerak bo'lganda yaratiladi (lazy). Shunda modulni import
qilishning o'zi .env yoki bazani talab qilmaydi — masalan, testlar bazasiz ishlaydi.
"""

from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.config import get_settings


@lru_cache
def get_engine() -> AsyncEngine:
    """Bazaga ulanishlar hovuzi (pool). Dastur bo'yi bitta bo'ladi."""
    return create_async_engine(get_settings().database_url)


@lru_cache
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    """Sessiya — bitta ish birligi (bir yoki bir nechta so'rov va commit)."""
    return async_sessionmaker(get_engine(), expire_on_commit=False)
