# O'rnatish qo'llanmasi (macOS, Linux, Windows)

Bu hujjat loyihani **ishlab chiqish** uchun kompyuterga nimalar o'rnatilishini
va qaysi tartibda o'rnatilishini tushuntiradi.

> Faqat **ishga tushirib ko'rish** kerak bo'lsa (masalan, tekshiruvchi uchun),
> Git va Docker yetarli bo'ladi: hamma narsa konteynerda ishlaydi.
> Bu yo'l loyiha oxirida README'da yoziladi.

## Umumiy ko'rinish

| # | Nima | Nega kerak | Qayerda ishlaydi |
|---|---|---|---|
| 1 | Git | Kodni saqlash, repo'ni topshirish | Kompyuterda |
| 2 | Python 3.12 yoki 3.13 | Butun kod Python'da | Kompyuterda |
| 3 | Poetry | Python kutubxonalari va virtual muhitni boshqarish | Kompyuterda |
| 4 | Docker | PostgreSQL va Qdrant'ni o'rnatmasdan ishga tushirish | Kompyuterda |
| 5 | Python kutubxonalari | torch, bge-m3, FastAPI, aiogram va boshqalar | Loyiha `.venv` ichida |
| 6 | bge-m3 modeli (~2.3 GB) | Matnni vektorga aylantirish | `models/` papkasida |
| 7 | PostgreSQL + Qdrant | Matnlar va vektorlar | Docker konteynerlarida |
| 8 | Telegram bot tokeni | Botni ulash | `.env` faylida |

**Tartib muhim:** 1–4 tizim dasturlari, 5–8 loyihaning o'zi. Har bir qadam
oldingisiga tayanadi: Poetry Python'siz o'rnatilmaydi, kutubxonalar Poetry'siz,
model esa kutubxonalarsiz yuklanmaydi.

**PostgreSQL va Qdrant kompyuterga alohida o'rnatilmaydi.** Ular Docker
konteynerlarida ishlaydi. Shu tufayli ularning versiyasi hammada bir xil bo'ladi,
o'chirish ham oson (`docker compose down`).

## Kompyuter talablari

| | Minimal | Tavsiya |
|---|---|---|
| RAM | 8 GB | 16 GB |
| Bo'sh disk | 10 GB | 15 GB |
| GPU | Shart emas | NVIDIA (Linux/Windows) yoki Apple Silicon (Mac) tezlashtiradi |
| Portlar | 5432 (PostgreSQL), 6333 (Qdrant), 8000 (API) bo'sh bo'lishi kerak | |

Diskdagi joy taxminan shunday taqsimlanadi: bge-m3 modeli ~2.3 GB, PyTorch va
boshqa kutubxonalar 1–2 GB, Docker image'lari ~1 GB.

> **Intel protsessorli Mac:** PyTorch 2.3 versiyasidan boshlab Intel Mac uchun
> chiqarilmaydi. Bunday kompyuterda loyihani Docker orqali ishga tushiring.

---

## 1. Git

**macOS**
```bash
xcode-select --install
```
Yoki Homebrew bo'lsa: `brew install git`.

**Linux (Ubuntu/Debian)**
```bash
sudo apt update && sudo apt install -y git
```

**Windows** (PowerShell)
```powershell
winget install --id Git.Git -e
```

Tekshirish: `git --version`

---

## 2. Python (3.12 yoki 3.13)

**Nega aynan shu versiyalar:** ikkalasi uchun ham PyTorch va boshqa kutubxonalar
tayyor holda chiqariladi. Ubuntu 24.04 da 3.12 standart o'rnatilgan, shuning
uchun loyiha ikkalasini qabul qiladi (`pyproject.toml` → `requires-python`).

**macOS**
```bash
brew install python@3.13
```
Yoki python.org saytidan o'rnatuvchini yuklab oling.

**Linux (Ubuntu 24.04 / Debian 13)**
```bash
sudo apt install -y python3 python3-venv python3-pip
```

**Windows**
```powershell
winget install --id Python.Python.3.13 -e
```
python.org o'rnatuvchisi bilan o'rnatsangiz, birinchi oynada
**"Add python.exe to PATH"** belgisini albatta qo'ying.

Tekshirish: `python3 --version` (Windows'da `py --version`)

---

## 3. Poetry

Poetry kutubxonalarni o'rnatadi, versiyalarini `poetry.lock` faylida qotiradi
va loyiha uchun alohida virtual muhit (`.venv`) yaratadi. Shu tufayli loyiha
har bir kompyuterda bir xil versiyalar bilan ishlaydi.

**macOS / Linux**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```
Keyin `~/.local/bin` papkasini PATH'ga qo'shing (o'rnatuvchi buni qanday
qilishni ekranda ko'rsatadi). macOS'da `brew install poetry` ham bo'ladi.

**Windows** (PowerShell)
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```
Keyin `%APPDATA%\Python\Scripts` papkasini PATH'ga qo'shing.

Tekshirish: `poetry --version` (2.x bo'lishi kerak)

---

## 4. Docker

**macOS**
[Docker Desktop](https://www.docker.com/products/docker-desktop/) ni o'rnating
(Apple Silicon yoki Intel versiyasini tanlang) yoki:
```bash
brew install --cask docker
```
O'rnatgandan keyin Docker ilovasini oching: u ishlab turishi kerak.

> **8 GB RAM'li Mac:** Docker Desktop → Settings → Resources → Memory ni
> **3 GB** atrofida qo'ying. PostgreSQL va Qdrant uchun shu yetadi, qolgan
> xotira bge-m3 modeliga kerak bo'ladi.

**Linux (Ubuntu/Debian)**
```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
```
Keyin tizimdan chiqib qayta kiring, shunda `docker` buyrug'i `sudo`'siz ishlaydi.
`docker compose` ham shu bilan birga o'rnatiladi.

**Windows**
1. Administrator sifatida ochilgan PowerShell'da WSL2 ni o'rnating va
   kompyuterni qayta ishga tushiring:
   ```powershell
   wsl --install
   ```
2. [Docker Desktop](https://www.docker.com/products/docker-desktop/) ni o'rnating.
   Settings bo'limida "Use the WSL 2 based engine" yoqilgan bo'lsin.

Tekshirish:
```bash
docker --version
docker compose version
docker run --rm hello-world
```

> **Windows uchun maslahat:** loyihani to'liq WSL2 (Ubuntu) ichida yuritish eng
> oson yo'l. Unda 1–3-qadamlarni "Linux" buyruqlari bilan WSL ichida bajarasiz
> va Windows'ga xos yo'l (PATH) muammolari bo'lmaydi.

---

## 5. Loyiha kutubxonalari

Loyiha papkasida:
```bash
poetry install
```

Bu buyruq `pyproject.toml` dagi hamma kutubxonani loyiha ichidagi `.venv`
papkasiga o'rnatadi. Kutubxonalar loyiha qadamlari bo'yicha qo'shib boriladi:

| Kutubxona | Vazifasi | Qaysi qadamda |
|---|---|---|
| `pandas` | Ma'lumotni o'rganish va tahlil qilish | 1 |
| `sqlalchemy`, `asyncpg`, `alembic` | PostgreSQL bilan ishlash, jadval migratsiyalari | 2 |
| `torch` | Neyron tarmoq dvigateli, bge-m3 shu ustida ishlaydi | 3 |
| `sentence-transformers` | bge-m3 ni yuklash va vektor hisoblash | 3 |
| `qdrant-client` | Qdrant bilan ishlash | 4 |
| `fastapi`, `uvicorn` | API server | 8 |
| `aiogram` | Telegram bot | 9 |
| `httpx` | Botdan API'ga so'rov yuborish | 9 |
| `pydantic-settings` | `.env` dan sozlamalarni o'qish | 2 |
| `pytest`, `ruff` | Testlar va kod sifatini tekshirish | barcha qadamlarda |

**GPU haqida:**
- Mac (Apple Silicon): GPU (MPS) avtomatik ishlatiladi.
- Linux'da NVIDIA karta bo'lsa, standart `torch` CUDA bilan keladi.
- Windows'da standart `torch` faqat CPU'da ishlaydi.

Bu topshiriq uchun CPU ham yetarli: 3048 ta qisqa savolni vektorga aylantirish
bir necha daqiqa oladi, bitta savol esa bir zumda ishlanadi.

---

## 6. bge-m3 modeli

Model Hugging Face'dan bir marta yuklab olinadi (~2.3 GB) va `models/bge-m3`
papkasida saqlanadi. Bu papka `.gitignore` da, gitga kirmaydi.

```bash
poetry run hf download BAAI/bge-m3 --local-dir models/bge-m3 --exclude "onnx/*" "imgs/*"
```

Keyin modelni bir marta fp16 (yarim aniqlik) ga o'giring:

```bash
poetry run python scripts/convert_model_fp16.py
```

Natija `models/bge-m3-fp16` papkasiga safetensors formatida saqlanadi (1.1 GB).
Tizim aynan shu nusxani ishlatadi. O'lchangan natijalar (Apple M, MPS):

| | Asl fp32 | fp16 nusxa |
|---|---|---|
| Doimiy xotira (RAM + GPU) | 2.55 GB | 1.58 GB |
| Yuklanish paytidagi eng yuqori RAM | 0.93 GB | 0.94 GB |
| Yuklanish vaqti | 13.6 s | 3.3 s |
| O'xshashlik baholaridagi farq | — | ≤ 0.0004, tartib bir xil |

GPU bo'lmagan kompyuterda (faqat CPU) kod fp32 da ishlaydi, chunki CPU'da fp16 sekin.

`hf` buyrug'i `huggingface_hub` kutubxonasi bilan birga keladi. U
`sentence-transformers` bilan avtomatik o'rnatiladi, alohida o'rnatish shart emas.

**Nega modelni alohida papkaga yuklaymiz:** yopiq (internetsiz) serverda model
internetdan yuklanmaydi. Papkani oldindan tayyorlab, serverga ko'chirish mumkin.
Kod modelni shu papkadan o'qiydi.

---

## 7. PostgreSQL va Qdrant (Docker'da)

Loyiha papkasida:
```bash
docker compose up -d
```

`docker-compose.yml` fayli 2-qadamda yaratiladi.

Tekshirish:
- `docker compose ps`: ikkala servis `running` holatida bo'lishi kerak.
- Qdrant'ning brauzerdagi paneli: http://localhost:6333/dashboard

---

## 8. Telegram bot tokeni

1. Telegram'da **@BotFather** ga yozing va `/newbot` buyrug'ini yuboring.
2. Bot nomini kiriting. BotFather sizga token beradi.
3. Tokenni loyihadagi `.env` fayliga yozing. `.env.example` fayli 2-qadamda
   yaratiladi, undan nusxa olib to'ldirasiz.

> Token parol bilan teng. Uni gitga, chatga yoki skrinshotga qo'ymang.
> `.env` fayli `.gitignore` da turibdi.

---

## 9. Hammasi joyidami: yakuniy tekshiruv

```bash
git --version
python3 --version
poetry --version
docker compose version
poetry run python -c "import torch; print('torch', torch.__version__)"
```

## Ko'p uchraydigan muammolar

| Belgi | Sababi | Yechim |
|---|---|---|
| `Cannot connect to the Docker daemon` | Docker ishlamayapti | Docker Desktop'ni oching (Linux'da: `sudo systemctl start docker`) |
| `port is already allocated` (5432) | Kompyuterda boshqa PostgreSQL ishlab turibdi | Uni to'xtating yoki `docker-compose.yml` dagi portni o'zgartiring |
| `poetry: command not found` | Poetry PATH'da yo'q | 3-qadamdagi PATH ko'rsatmasini bajaring |
| Model yuklanayotganda xotira yetmayapti | RAM kam | Boshqa dasturlarni yoping, Docker xotirasini kamaytiring |
| Windows'da `python` Microsoft Store'ni ochadi | Windows'ning "app alias" funksiyasi | `py` buyrug'ini ishlating yoki Settings → App execution aliases dan o'chiring |
