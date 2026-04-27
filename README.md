# 🐂🐑🐎 MalChain

> **Әр мал — паспорт. Әр паспорт — мүмкіндік.**
> _Every animal — a passport. Every passport — an opportunity._

[![Status](https://img.shields.io/badge/status-MVP-blue)]()
[![Hackathon](https://img.shields.io/badge/CyberShield-2026-orange)](https://ku.edu.kz/hackathon)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Made in](https://img.shields.io/badge/made%20in-Kazakhstan-blue)]()
[![Python](https://img.shields.io/badge/python-3.11-yellow)]()

---

## 🇰🇿 Қысқаша

**MalChain** — Қазақстанның мал шаруашылығы саласына арналған ИИ платформасы.
Әр жануарға (ірі қара, қой, жылқы) биометриялық суреттер арқылы цифрлық
паспорт береді және субсидия жүйесінде қайталанулардың алдын алады.

### 🧩 Қандай мәселені шешеді?

1. **Субсидия алаяқтығы** — бір жануар бірнеше иесінің атынан тіркеліп, бюджеттен
   қайталап ақша алуға болады.
2. **Тіркеу сенімсіздігі** — қазіргі бирка/чип жүйелері бұзылады, ауыстырылады.
3. **Аудиторлардың құралсыздығы** — далада жануарды жылдам тексеру мүмкіндігі шектеулі.

### 💡 Шешім

Үш сатылы көру жүйесі: **YOLOv8 → ResNet50 → FAISS**.

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ 1. YOLOv8n      │ --> │ 2. ResNet50      │ --> │ 3. FAISS         │
│  Species classf │     │  512-d embedding │     │  per-species     │
│  cattle/sheep/  │     │  L2-normalised   │     │  IndexFlatIP     │
│  horse + bbox   │     │                  │     │  cosine search   │
└─────────────────┘     └──────────────────┘     └──────────────────┘
```

> ⚠️ MVP: используется ResNet50 ImageNet (zero-shot) с детерминированной
> seeded-проекцией 2048 → 512. Production-вариант — fine-tuning на датасетах
> муздықтары/мордочек с triplet loss.

---

## 🛠 Tech stack

| Layer | Tool | Why |
|---|---|---|
| Detection | **Ultralytics YOLOv8n** | COCO already has cow/sheep/horse |
| Features | **PyTorch + ResNet50** | Strong pretrained backbone |
| Vector search | **FAISS (IndexFlatIP)** | Cosine via inner product, per-species |
| API | **FastAPI 0.109** | Async + auto OpenAPI |
| ORM | **SQLAlchemy 2.0** | Modern declarative |
| Validation | **Pydantic v2 + pydantic-settings** | Schemas + .env |
| DB | **SQLite** | Zero-setup |
| Frontend | **Streamlit 1.30** | Multipage rapid UI |
| Container | **Docker / compose** | Reproducible |

### ⚡ Performance Core (Rust + PyO3)

MalChain now includes an optional native acceleration layer:

- Rust workspace: `rust_core/`
- PyO3 extension: `rust_core/crates/agroai_core`
- Python facade with graceful fallback: `app/core/accel.py`

If the Rust extension is installed, hot paths (image decode/resize/normalize,
hashing, NMS, vector similarity helpers) run through `agroai_core`.
If it is missing, the app automatically falls back to pure Python/NumPy/OpenCV.

Build and run:

```bash
make rust-build
make rust-install
make rust-test
make rust-bench
```

See [`docs/RUST_CORE.md`](docs/RUST_CORE.md) for architecture and integration details.

---

## 🚀 Quickstart

```bash
git clone https://github.com/<org>/malchain.git
cd malchain
python -m venv .venv && source .venv/bin/activate
make install
make download-models
make -j2 run               # backend :8000  +  frontend :8501
```

Open:
- **API docs** — http://localhost:8000/docs
- **Streamlit UI** — http://localhost:8501

### Docker

```bash
make docker-build
make docker-up
```

---

## 📂 Project structure

```
malchain/
├── app/                      # FastAPI backend
│   ├── api/                  # Routers (animals, stats)
│   ├── ai/                   # YOLO + ResNet50 + pipeline
│   ├── database/             # SQLAlchemy + FAISS wrapper
│   ├── schemas/              # Pydantic models
│   ├── utils/                # Image & ID helpers
│   ├── config.py
│   ├── exceptions.py
│   └── main.py
├── frontend/                 # Streamlit multipage app
├── ml_models/                # Weights (downloaded at runtime)
├── data/vector_dbs/          # FAISS indices (per species)
├── scripts/                  # download_models, seed_demo_data
├── tests/                    # pytest
├── docs/                     # ARCHITECTURE.md, API.md
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── requirements.txt
```

---

## 🌐 API summary

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness |
| `POST` | `/api/animals/identify` | Identify animal from photo |
| `POST` | `/api/animals/register` | Register new animal |
| `GET` | `/api/animals/{id}` | Animal details |
| `GET` | `/api/animals/by-owner/{iin}` | All animals of an owner |
| `DELETE` | `/api/animals/{id}` | Remove |
| `GET` | `/api/stats` | Per-species stats |

### Example — identify

```bash
curl -X POST http://localhost:8000/api/animals/identify \
  -F "photo=@cow.jpg"
```

```json
{
  "status": "matched",
  "species": "cattle",
  "detection_confidence": 0.94,
  "best_match": { "animal_id": "COW-A7F8B2", "similarity": 0.91 }
}
```

---

## 🎬 Demo flow

1. **🆕 Register** — upload `cow1.jpg` for owner `880101300123` → returns `COW-XXXXXX`.
2. **🔍 Identify** — upload the same image → status `matched`, similarity ≥ 0.85.
3. **🆕 Register again** → backend rejects with `409 duplicate_animal`.
4. **🔍 Identify** a different animal → status `new`.

---

## 🗺 Roadmap

- ✅ **MVP (this PR)** — species detection + biometric registration + duplicate guard
- 🔜 **Hackathon polish** — demo dataset, presentation, fine-tuning notebook
- 📅 **Pilot (3 months)** — fine-tune muzzle / face heads with triplet loss
- 📅 **National (12 months)** — ИСЖ integration, audit module
- 📅 **Regional (24 months)** — Central Asia roll-out

---

## 🗂 Datasets

MalChain only references **publicly-verified** livestock datasets — no
fabricated URLs, no fabricated metrics. The manifest is at
[`data/datasets.yaml`](data/datasets.yaml) and the CLI driver at
[`scripts/download_datasets.py`](scripts/download_datasets.py).

```bash
make install-dev          # adds PyYAML, datasets, pytorch-metric-learning, jupyterlab
make download-datasets    # = python scripts/download_datasets.py --list
make dataset-status       # show ready/manual-pending status per dataset
make dataset-validate     # validate payloads + format checks (image probe / YOLO / split paths)
make eval-reid-baseline   # baseline re-id metrics on identity folders (override DATA_ROOT=...)
python scripts/download_datasets.py --download cattle   # huggingface / git modes auto-fetch
python scripts/download_datasets.py --download all      # manual modes print instructions
```

See:

- [`docs/DATASETS.md`](docs/DATASETS.md) — full per-dataset notes, licensing,
  privacy, and citation template (with `// TODO` placeholders, never invented
  citations).
- [`docs/NEXT_STEPS.md`](docs/NEXT_STEPS.md) — prioritized operational backlog
  for completing remaining manual datasets and model evaluation.
- [`docs/MODELS.md`](docs/MODELS.md) — honest framing of YOLOv8n + ResNet50
  (zero-shot baseline; accuracy not yet measured for this repo) and the
  fine-tuning roadmap via [`notebooks/finetune_baseline.ipynb`](notebooks/finetune_baseline.ipynb).

API keys (`ROBOFLOW_API_KEY`, `HUGGINGFACE_HUB_TOKEN`, `KAGGLE_KEY`) are read
from your environment — **never commit them**.

---

## 🧪 Tests

```bash
make test
```

Pipeline & API are covered with mocked ML (no real YOLO/ResNet50 in unit tests).

---

## 🏆 Hackathon

Built for **CyberShield Hackathon 2026** — Kozybayev University, Petropavl —
[ku.edu.kz/hackathon](https://ku.edu.kz/hackathon).
Track: _AI in Agri-Analytics_.

## 👥 Team

> _Add your names, roles and contacts here._

## 📄 License

MIT — see [LICENSE](LICENSE).

---

## 🇬🇧 English summary

MalChain is a livestock biometric identification MVP. It classifies cattle /
sheep / horse with YOLOv8, extracts a 512-dim embedding via ResNet50, and
matches it against per-species FAISS indices. Built in 24h for the CyberShield
2026 hackathon. Features: register, identify, duplicate detection, owner
lookup, statistics dashboard. Bilingual UI (Kazakh primary). Production
roadmap: fine-tune extractors with triplet loss on muzzle/face datasets.
# agro-ai
