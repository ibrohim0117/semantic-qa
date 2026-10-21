"""bge-m3 tokenizatori so'zlarni qanday bo'lishini ko'rsatuvchi demo.

Algoritm: Unigram. Lug'atdagi har bir tokenning bali bor (log-ehtimollik).
So'zni bo'lishning barcha mumkin variantlari ichidan ballari yig'indisi
eng katta bo'lgani tanlanadi.
Ishga tushirish: poetry run python scripts/demo_tokenizer.py
"""

import json
import math

from tokenizers import Tokenizer

TOKENIZER_PATH = "models/bge-m3/tokenizer.json"

WORD = "salom"  # batafsil tahlil qilinadigan so'z
WORDS = [
    "salom",
    "hello",
    "привет",
    "qanday",
    "pasportimni",
    "almashtiraman",
    "almashtrsh",
    "o'g'il",
    "o‘g‘il",
    "oʻgʻil",
]


def all_splits(text: str, vocab: dict[str, float]) -> list[list[str]]:
    """Matnni lug'atdagi bo'laklarga bo'lishning barcha variantlari."""
    if not text:
        return [[]]
    result = []
    for end in range(1, len(text) + 1):
        piece = text[:end]
        if piece in vocab:
            for rest in all_splits(text[end:], vocab):
                result.append([piece, *rest])
    return result


def main() -> None:
    with open(TOKENIZER_PATH, encoding="utf-8") as f:
        data = json.load(f)
    vocab = {token: score for token, score in data["model"]["vocab"]}
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    # 1. Bitta so'zni bo'lishning barcha variantlari va ularning ballari
    text = "▁" + WORD  # "▁" = so'z boshi (oldida bo'shliq bor)
    splits = all_splits(text, vocab)
    ranked = sorted(splits, key=lambda s: -sum(vocab[p] for p in s))

    print(f"1) «{WORD}» ni bo'lishning {len(splits)} ta varianti bor. Eng yaxshi 6 tasi:\n")
    for parts in ranked[:6]:
        scores = [vocab[p] for p in parts]
        detail = " + ".join(f"{p}({s:.2f})" for p, s in zip(parts, scores, strict=True))
        print(f"   {sum(scores):8.2f}  =  {detail}")

    best = ranked[0]
    real = tokenizer.encode(WORD, add_special_tokens=False).tokens
    print(f"\n   Hisob bo'yicha g'olib:    {best}")
    print(f"   Tokenizator haqiqatda:    {real}")

    # Ball nimani anglatadi: ehtimollik
    for p in best:
        print(f"   «{p}» bali {vocab[p]:.2f} → ehtimolligi ≈ {math.exp(vocab[p]):.6f}")

    # 2. Turli so'zlar qanday bo'linadi
    print("\n2) Turli so'zlar:\n")
    for w in WORDS:
        tokens = tokenizer.encode(w, add_special_tokens=False).tokens
        print(f"   {w:<15} → {len(tokens)} ta: {tokens}")


if __name__ == "__main__":
    main()
