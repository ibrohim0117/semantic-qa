"""Loyiha sozlamalari: .env faylidan va muhit o'zgaruvchilaridan o'qiladi.

Majburiy qiymat yo'q bo'lsa, dastur ishga tushishdayoq aniq xato bilan to'xtaydi
(fail fast). Shunda xato keyinroq, bazaga ulanayotganda, tushunarsiz holda chiqmaydi.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    model_path: str = "models/bge-m3-fp16"

    @property
    def database_url(self) -> URL:
        # URL.create parolda maxsus belgilar (@, :, /) bo'lsa ham to'g'ri ishlaydi
        return URL.create(
            "postgresql+asyncpg",
            username=self.postgres_user,
            password=self.postgres_password,
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        )


@lru_cache
def get_settings() -> Settings:
    """Sozlamalarni bir marta o'qiydi va keyin o'sha obyektni qaytaradi."""
    return Settings()
