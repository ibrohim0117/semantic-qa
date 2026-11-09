# Semantic QA — ma'no bo'yicha savol-javob qidiruvi

Davlat xizmatlari bo'yicha savollarga **tayyor, tekshirilgan javobni** ma'no bo'yicha topib beradigan tizim.
Foydalanuvchi savolni o'z so'zi bilan yozadi, tizim bazadagi eng yaqin savollarni topadi va shu savolga
biriktirilgan javobni **so'zma-so'z** qaytaradi.

AKTRM AI/ML lavozimining test topshirig'i sifatida qilingan. Ma'lumot: **505 javob** va **3 048 savol varianti**.

> **Holat: jarayonda.** Ma'lumot yuklash, vektorlash va qidiruv tayyor. Yakuniy baholash, «topilmadi»
> chegarasini tanlash, FastAPI va Telegram bot hali qilinmagan (pastda «Keyingi qadamlar»).

## Arxitektura

```
Indekslash:  CSV ──► tekshirish ──► PostgreSQL ──► normalize ──► bge-m3 ──► Qdrant

Qidiruv:     savol ──► normalize ──► bge-m3 ──► Qdrant (cosine) ──► javob bo'yicha guruhlash
                                                                          │
                                                       PostgreSQL'dan tayyor javob ◄┘
```

| Texnologiya | Vazifasi |
|---|---|
| bge-m3 (1024 o'lcham) | matnni vektorga aylantirish, lokal |
| Qdrant | vektor qidiruv |
| PostgreSQL | savol va javoblarning asosiy manbai |
| SQLAlchemy, Alembic | baza va migratsiyalar |

## Asosiy qarorlar

- **LLM ishlatilmaydi.** Javoblar tayyor va tekshirilgan, vazifa ularni yozish emas, topish. Shuning uchun
  gallyutsinatsiya xavfi yo'q.
- **Vektorga javob emas, savollar aylantiriladi.** Foydalanuvchi savoli javob matniga emas, boshqa savolga
  o'xshaydi.
- **PostgreSQL — asosiy manba, Qdrant — undan yasalgan indeks.** Indekslash idempotent: nuqta ID'si savol
  ID'siga teng.
- **Matn bitta funksiyada tozalanadi.** Tokenizator `o'g'il`, `o‘g‘il` va `oʻgʻil`ni uch xil tokenlarga
  bo'ladi, shuning uchun apostroflar indekslashda ham, so'rovda ham bir xil qilinadi.
- **Model fp16 ga o'girilgan.** 8 GB RAM'li kompyuter uchun: doimiy xotira 2,55 → 1,58 GB, ballar farqi
  ≤ 0,0004, natijalar tartibi o'zgarmagan.

## O'lchovlar

- Vektorlash ~39 ms, Qdrant qidiruvi ~3 ms. Vaqtning 90% dan ko'pi modelga ketadi.
- HNSW natijasi aniq qidiruv bilan solishtirilgan: top-10 100% bir xil.
- 34 ta test (`pytest`).

## Sifat qanday tekshiriladi

Uchta alohida to'plam ishlatiladi:

| To'plam | Nima | Natija |
|---|---|---|
| **A.** Ajratilgan variantlar | har javobdan 1 savol indeksdan chiqarilgan | recall@1 = 99,6% — **optimistik** |
| **B.** Mustaqil yozilgan savollar (`eval/test_b.csv`, 76 ta) | so'zlashuv tili, imlo xatolari, kirill, rus tili | hali o'lchanmagan |
| **C.** Doiradan tashqari savollar (`eval/test_c.csv`, 37 ta) | kutilgan natija — «topilmadi» | hali o'lchanmagan |

**Nega A natijasiga ishonib bo'lmaydi:** savollarning 5/6 qismi shablondan yasalgan. Bitta asl savolning
variantlari bir-biriga juda o'xshaydi, ya'ni yashirilgan savolning «egizagi» bazada qoladi. Shuning uchun A faqat
texnik tekshiruv. Asosiy o'lchov — B to'plami.

**«Topilmadi» chegarasi o'lchab tanlanishi kerak:** aloqasiz savollar ham eng yaqin javobga 0,42–0,68 ball oladi.
Vektor qidiruv doim nimadir qaytaradi.

## Ishga tushirish

Talablar: Python 3.12–3.13, Poetry, Docker. Batafsil: [docs/ORNATISH.md](docs/ORNATISH.md).

```bash
cp .env.example .env
poetry install
docker compose up -d

# Model (~2.3 GB) va uning fp16 nusxasi
poetry run hf download BAAI/bge-m3 --local-dir models/bge-m3 --exclude "onnx/*" "imgs/*"
poetry run python scripts/convert_model_fp16.py

# Baza va indeks
poetry run alembic upgrade head
poetry run python -m src.ingest
poetry run python -m src.index_questions

# Sinab ko'rish
poetry run python -m scripts.qdrant_search_demo
poetry run pytest
```

## Cheklovlar

- B va C to'plamlaridagi yakuniy natija hali yo'q.
- Imlo xatosi ballni keskin tushiradi: xatoli savol 0,65, boshqa javobga tegishli savol 0,63 oladi.
- Ma'lumotdagi ba'zi javoblar bir-biriga juda yaqin, bunday holatda birinchi o'rin tasodifiy bo'lishi mumkin.

## Keyingi qadamlar

1. B va C to'plamlarida recall@k, MRR va «topilmadi» aniqligini o'lchash, chegarani tanlash.
2. Natija yetmasa — gibrid qidiruv yoki reranker qo'shish.
3. FastAPI (`POST /ask`) va Telegram bot.
4. To'liq Docker Compose.
