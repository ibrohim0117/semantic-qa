"""Matnni tozalash (normalizatsiya).

Indekslashda ham, foydalanuvchi savolida ham AYNAN shu funksiya ishlatiladi,
aks holda ikki tomon "har xil tilda" bo'lib qoladi. Faqat dalili bor o'zgartirishlar
qilinadi; kichik harfga o'tkazish, kirill->lotin, imlo tuzatish ataylab yo'q.
"""

import re
import unicodedata

# O'zbekcha o' va g' harflarida uchraydigan apostrof turlari. Tokenizator ularni
# har xil token deb ko'radi (scripts/demo_tokenizer.py), bazada esa faqat ASCII ' bor.
APOSTROPHES = re.compile("[‘’ʻʼʹ`´]")  # ‘ ’ ʻ ʼ ʹ ` ´
SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    # NFC: ikki belgidan yasalgan harflarni bitta belgiga keltiradi (у + ˘ -> ў)
    text = unicodedata.normalize("NFC", text)
    text = APOSTROPHES.sub("'", text)
    return SPACES.sub(" ", text).strip()
