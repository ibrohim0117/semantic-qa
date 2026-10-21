"""bge-m3 ni bir marta fp16 ga o'girib, safetensors formatida saqlash.

Nega: asl pytorch_model.bin fp32 da. Uni har safar fp16 ga o'girish yuklanish
paytida RAM'ni ~3 GB gacha ko'taradi (8 GB'li Mac'da swap'ga olib keladi).
fp16 nusxa diskdan to'g'ridan-to'g'ri o'qiladi va o'girish kerak bo'lmaydi.
safetensors faylida kod bo'lmaydi (pickle'dan farqli), shuning uchun xavfsizroq.

Ishga tushirish (bir marta, modelni yuklab olgandan keyin):
    poetry run python scripts/convert_model_fp16.py
"""

import torch
from sentence_transformers import SentenceTransformer

SOURCE = "models/bge-m3"
TARGET = "models/bge-m3-fp16"


def main() -> None:
    model = SentenceTransformer(SOURCE, device="cpu", model_kwargs={"dtype": torch.float16})
    model.save(TARGET, safe_serialization=True)
    print(f"Saqlandi: {TARGET}")


if __name__ == "__main__":
    main()
