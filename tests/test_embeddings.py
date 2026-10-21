"""src/embeddings.py testlari.

Tez testlar modelsiz ishlaydi. "slow" testlar haqiqiy modelni (1.1 GB) talab qiladi;
model yo'q kompyuterda (masalan CI'da) ular avtomatik o'tkazib yuboriladi.
Faqat tez testlar: poetry run pytest -m "not slow"
"""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src import embeddings
from src.embeddings import Embedder, pick_device

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/bge-m3-fp16"))
requires_model = pytest.mark.skipif(not MODEL_PATH.exists(), reason=f"model yo'q: {MODEL_PATH}")


# --- Tez testlar (modelsiz) -------------------------------------------------


def test_pick_device_returns_known_value():
    assert pick_device() in {"cuda", "mps", "cpu"}


def test_missing_model_gives_clear_error():
    with pytest.raises(FileNotFoundError, match="docs/ORNATISH.md"):
        Embedder("models/bunday-model-yoq")


def test_get_embedder_creates_model_only_once(monkeypatch):
    # Haqiqiy model o'rniga soxta klass (mock): u faqat necha marta yaratilganini yozadi
    created = []

    class FakeEmbedder:
        def __init__(self, model_path):
            created.append(model_path)

    monkeypatch.setattr(embeddings, "Embedder", FakeEmbedder)
    monkeypatch.setattr(embeddings, "get_settings", lambda: SimpleNamespace(model_path="x"))
    embeddings.get_embedder.cache_clear()
    try:
        first, second = embeddings.get_embedder(), embeddings.get_embedder()
        assert first is second
        assert created == ["x"]
    finally:
        embeddings.get_embedder.cache_clear()


# --- Sekin testlar (haqiqiy model) -------------------------------------------


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    """Model shu fayldagi barcha testlar uchun bir marta yuklanadi."""
    return Embedder(MODEL_PATH)


@pytest.mark.slow
@requires_model
def test_output_shape_type_and_unit_length(embedder):
    vectors = embedder.encode(["ID-karta necha kunda tayyor bo'ladi?", "salom", ""])
    assert vectors.shape == (3, embedder.dim) == (3, 1024)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)


@pytest.mark.slow
@requires_model
def test_apostrophe_variants_give_same_vector(embedder):
    a, b, c = embedder.encode(["o‘g‘lim bog‘chada", "oʻgʻlim bogʻchada", "o'g'lim bog'chada"])
    assert np.allclose(a, c) and np.allclose(b, c)


@pytest.mark.slow
@requires_model
def test_paraphrase_is_closer_than_unrelated(embedder):
    query, paraphrase, unrelated = embedder.encode(
        [
            "pasportimni qanday almashtiraman?",
            "Pasportni (ID-karta) almashtirish tartibi qanday?",
            "Ertaga yomg'ir yog'adimi?",
        ]
    )
    assert query @ paraphrase > query @ unrelated + 0.2


@pytest.mark.slow
@requires_model
def test_encode_one_matches_encode(embedder):
    text = "soliqni qayerdan to'layman?"
    assert np.allclose(embedder.encode_one(text), embedder.encode([text])[0])
