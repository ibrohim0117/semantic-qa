"""src/text.py testlari: normalize() faqat kerakli narsani o'zgartiradi."""

import pytest

from src.text import normalize


@pytest.mark.parametrize("variant", ["o‘g‘il", "o’g’il", "oʻgʻil", "oʼgʼil", "o`g`il", "o´g´il"])
def test_apostrophe_variants_become_ascii(variant):
    assert normalize(variant) == "o'g'il"


def test_extra_whitespace_is_collapsed():
    assert normalize("  ID   karta \t\n yo'qoldi  ") == "ID karta yo'qoldi"


def test_decomposed_letter_is_composed():
    decomposed = "ў"  # у + qo'shiluvchi "˘" = ekranda "ў", lekin 2 ta belgi
    assert len(decomposed) == 2
    assert normalize(decomposed) == "ў"
    assert len(normalize(decomposed)) == 1


def test_case_and_cyrillic_are_kept():
    # Kichik harfga o'tkazish va kirill->lotin ataylab qilinmaydi
    assert normalize("ID-Karta Паспорт") == "ID-Karta Паспорт"


def test_normalize_is_idempotent():
    text = " o‘g‘lim  bog‘chaga "
    assert normalize(normalize(text)) == normalize(text)
