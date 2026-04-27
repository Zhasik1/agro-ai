# MalChain — Architecture

## 1. Component overview

```
┌─────────────────────────────┐         ┌──────────────────────────────┐
│        Streamlit UI         │ ──HTTP─▶│         FastAPI API          │
│  app.py + 3 pages           │  JSON   │  /api/animals/* + /api/stats │
└─────────────────────────────┘         └──────────────┬───────────────┘
                                                       │
                                ┌──────────────────────┼─────────────────────┐
                                ▼                      ▼                     ▼
                       ┌────────────────┐    ┌──────────────────┐   ┌─────────────────┐
                       │  YOLOv8n       │    │  ResNet50 + L2   │   │  SQLite (ORM)   │
                       │  species detect│    │  512-d embedding │   │  Animal / Owner │
                       └────────────────┘    └────────┬─────────┘   └─────────────────┘
                                                      ▼
                                          ┌────────────────────────┐
                                          │  FAISS IndexFlatIP     │
                                          │  cattle / sheep / horse │
                                          └────────────────────────┘
```

## 2. Data flows

### 2.1 `POST /api/animals/identify`

```
client → FastAPI → validate_image_bytes → decode_image → pipeline.identify_animal
                                                           ├─ classifier.detect()   ← YOLOv8n (species + bbox)
                                                           ├─ _sub_detect()         ← geometric ROI crop (muzzle / face)
                                                           ├─ extractor.extract()   ← ResNet50 → 512-d embedding
                                                           ├─ vector_db.search()
                                                           └─ threshold → status
```

Status mapping:
| Top similarity | Status |
|---|---|
| ≥ `MATCH_THRESHOLD` (0.85) | `matched` |
| ≥ `SUSPECT_THRESHOLD` (0.70) | `suspect` |
| else / no records | `new` |

### 2.2 `POST /api/animals/register`

Same first 4 steps as `identify`, then:
- if `matched` → `409 duplicate_animal`
- otherwise → generate id (`COW-/SHP-/HRS-XXXXXX`), `vector_db.add(...)`, persist
  the `Animal` row + upsert the `Owner`.

## 3. Why FAISS — and why per-species?

- `IndexFlatIP` over L2-normalised vectors gives exact cosine similarity.
- Splitting by species:
  1. eliminates cross-species false matches (a sheep can never match a cow);
  2. keeps each index small → sub-millisecond search at hackathon scale;
  3. simplifies deletion (rebuild only one species' index).

## 4. Storage layout

```
data/
├── malchain.db              # SQLite (Animal, Owner)
└── vector_dbs/
    ├── cattle/{index.faiss,ids.pkl}
    ├── sheep/{index.faiss,ids.pkl}
    └── horse/{index.faiss,ids.pkl}
```

All FAISS state is rebuildable from SQLite if embeddings are re-run; the
`Animal.photo_hash` column ensures deduplication at the byte level even
before vector search.

## 5. Concurrency

- A `threading.Lock` per species index serialises FAISS writes.
- FastAPI is run by a single uvicorn worker by default; for multi-worker
  deployments switch to a process-shared store (e.g. Qdrant) — see roadmap.

## 6. Future roadmap

- **Fine-tuned heads** — replace seeded random projection with a trained
  triplet-loss head per species.
- **Subsidy audit** — anomaly detection on owner-IIN density / region.
- **Satellite verification** — cross-check declared herd size with NDVI /
  thermal imagery.
- **National rollout** — migrate from SQLite/FAISS to Postgres + Qdrant.
- **Mobile capture** — on-device cropping, offline biometric capture.
