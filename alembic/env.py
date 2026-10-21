"""Alembic migratsiyalarini ishga tushiruvchi fayl.

Baza manzili alembic.ini dan emas, .env dan (src/config.py orqali) olinadi,
shunda parol hech qachon gitga tushmaydi va manzil bitta joyda turadi.
"""

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context
from src.config import get_settings
from src.db.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Bizning jadvallar chizmasi. --autogenerate shu bilan bazani solishtirib,
# farqini migratsiya fayliga o'zi yozadi.
target_metadata = Base.metadata
DATABASE_URL = get_settings().database_url


def run_migrations_offline() -> None:
    """Bazaga ulanmasdan, faqat SQL matnini chiqarish (alembic upgrade --sql)."""
    context.configure(
        url=DATABASE_URL.render_as_string(hide_password=False),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Bazaga ulanib, migratsiyalarni bitta tranzaksiyada bajarish."""
    engine = create_async_engine(DATABASE_URL, poolclass=pool.NullPool)
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_async_migrations())
