"""Matnni vektorga aylantirish (bge-m3).

Model jarayon bo'yi bir marta yuklanadi (get_embedder). Har bir matn vektorga
aylanishdan oldin normalize() dan o'tadi — indekslash ham, foydalanuvchi savoli ham
aynan shu yo'ldan o'tgani uchun ikki tomon doim bir xil tozalanadi.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.config import get_settings
from src.text import normalize


def pick_device() -> str:
    """Eng tez mavjud qurilma: NVIDIA GPU -> Apple GPU (MPS) -> CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Embedder:
    def __init__(self, model_path: str | Path, device: str | None = None) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model topilmadi: {path}. Yuklab olish va fp16 ga o'girish: "
                "docs/ORNATISH.md, 6-bo'lim."
            )
        self.device = device or pick_device()
        # GPU'da fp16: xotira ~2 barobar kam, baholar farqi <= 0.0004 (o'lchangan).
        # CPU'da fp16 sekin, shuning uchun u yerda fp32.
        self.dtype = torch.float16 if self.device != "cpu" else torch.float32
        self.model = SentenceTransformer(
            str(path), device=self.device, model_kwargs={"dtype": self.dtype}
        )
        self.dim = self.model.get_embedding_dimension()

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Matnlar -> (n, dim) float32 vektorlar, har birining uzunligi 1."""
        clean = [normalize(text) for text in texts]
        vectors = self.model.encode(clean, batch_size=batch_size, convert_to_numpy=True)
        # Uzunlikni 1 ga float32 da keltiramiz: fp16 da keltirilsa yaxlitlash sababli
        # ~1.0005 chiqadi, ko'paytmalar yig'indisi esa aniq cosine bo'lmay qoladi.
        vectors = vectors.astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-12, None)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


@lru_cache
def get_embedder() -> Embedder:
    """Modelni bir marta yuklaydi va keyin o'sha obyektni qaytaradi (~3 s, ~1.6 GB)."""
    return Embedder(get_settings().model_path)
