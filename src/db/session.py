"""Bazaga ulanish: bitta engine va undan sessiyalar yaratuvchi fabrika."""

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.config import get_settings

# Engine — bazaga ulanishlar hovuzi (pool). Dastur bo'yi bitta bo'ladi.
engine = create_async_engine(get_settings().database_url)

# Sessiya — bitta ish birligi (bir yoki bir nechta so'rov va commit).
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)
