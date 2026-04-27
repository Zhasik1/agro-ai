# 🗺 MalChain — Roadmap

Cross-references:
- [docs/NEXT_STEPS.md](NEXT_STEPS.md) — operational backlog (bugs, infra, DevOps)
- [docs/EVALUATION.md](EVALUATION.md) — planned accuracy benchmarks *(not yet created)*

---

## MVP (current)

- [x] YOLOv8n species detector (cattle / sheep / horse) with COCO weights
- [x] ResNet50 + seeded projection → 512-d L2-normalised embeddings
- [x] Per-species FAISS IndexFlatIP with cosine similarity thresholds
- [x] FastAPI backend: identify, register, delete, by-owner, stats endpoints
- [x] Streamlit multipage frontend (register / identify / statistics)
- [x] SQLite + SQLAlchemy ORM (Animal, Owner models)
- [x] Geometric muzzle / face sub-detector (deterministic ROI heuristic)
- [x] Python CI: ruff + black + pytest --cov=app

---

## Hackathon Day

- [ ] Seed demo dataset (≥ 30 images, 3 species, 2 owners) via `scripts/seed_demo_data.py`
- [ ] Live demo on Kazakh livestock photos presented at Kozybayev University
- [ ] Docker Compose one-command start: `docker compose up`
- [ ] README quickstart verified on a clean Ubuntu 22.04 machine
- [ ] Slide deck linking to live API docs at `/docs`

---

## Pilot (3 months)

- [ ] Fine-tune muzzle detector YOLO head on BovineNetID / Roboflow muzzle dataset
- [ ] Fine-tune face detector for sheep / horse on dedicated datasets
- [ ] Replace seeded random projection with triplet-loss-trained head (ResNet50 backbone)
- [ ] Evaluate baseline vs fine-tuned heads on held-out gallery/query splits
- [ ] Publish results to `docs/EVALUATION.md` with reproducible eval script
- [ ] Pilot deployment with 3–5 farms in North Kazakhstan

---

## National (12 months)

- [ ] Integrate with ИСЖ (Information System for Livestock) via REST adapter
- [ ] Migrate from SQLite / FAISS to PostgreSQL + Qdrant for multi-worker scale
- [ ] Subsidy audit module: anomaly detection on owner-IIN density / region
- [ ] Mobile capture app (Android/iOS): offline biometric capture + sync
- [ ] Role-based access control (farmer / inspector / admin)
- [ ] Cover all three major livestock species in ≥ 5 oblasts

---

## Regional (24 months)

- [ ] Satellite cross-verification: cross-check declared herd size with NDVI / thermal imagery
- [ ] Multi-animal frames: identify / register multiple animals per photo
- [ ] Cross-border tracking integration (EAEU livestock registry)
- [ ] On-device inference (ONNX / CoreML export) for fully offline field inspections
- [ ] Open dataset contribution: publish anonymised muzzle/face dataset for research
- [ ] Reach 95 % top-1 identification accuracy on held-out test sets for all three species
