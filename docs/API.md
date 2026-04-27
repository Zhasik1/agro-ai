# MalChain — REST API

Base URL: `http://localhost:8000`.
OpenAPI / Swagger UI: `/docs`. ReDoc: `/redoc`.

## Errors

All domain errors follow RFC 7807-ish JSON:

```json
{
  "type": "duplicate_animal",
  "title": "DuplicateAnimalError",
  "status": 409,
  "detail": "Бұл жануар тіркелген / Animal already registered",
  "existing_id": "COW-A7F8B2",
  "similarity": 0.93
}
```

| Code | HTTP | Meaning |
|---|---|---|
| `invalid_image` | 400 | Empty / oversized / wrong MIME |
| `animal_not_detected` | 422 | YOLO did not find cattle/sheep/horse |
| `duplicate_animal` | 409 | Same animal already registered |
| `animal_not_found` | 404 | No row for given id |
| `owner_not_found` | 404 | No owner for given IIN |

## Endpoints

### `GET /health`
```bash
curl http://localhost:8000/health
# {"status":"ok","version":"0.1.0"}
```

### `POST /api/animals/identify`
```bash
curl -X POST http://localhost:8000/api/animals/identify \
  -F "photo=@cow.jpg"
```

Response (`200`):
```json
{
  "status": "matched",
  "species": "cattle",
  "detection_confidence": 0.94,
  "bbox": [120, 80, 510, 460],
  "candidates": [{"animal_id": "COW-A7F8B2", "similarity": 0.91}],
  "best_match": {"animal_id": "COW-A7F8B2", "similarity": 0.91}
}
```

### `POST /api/animals/register`
```bash
curl -X POST http://localhost:8000/api/animals/register \
  -F "photo=@cow.jpg" \
  -F "owner_iin=880101300123" \
  -F "age_years=4" \
  -F "weight_kg=420" \
  -F "breed=Қазақтың ақбас сиыры"
```

Success (`201`):
```json
{
  "animal": {
    "id": "COW-A7F8B2",
    "species": "cattle",
    "owner_iin": "880101300123",
    "status": "active",
    "registered_at": "2026-04-29T10:15:00Z",
    "...": "..."
  },
  "detection_confidence": 0.94,
  "message": "Тіркеу сәтті аяқталды"
}
```

Duplicate (`409`): see error block above.
Validation (`422`): pydantic error array.

### `GET /api/animals/{animal_id}`
```bash
curl http://localhost:8000/api/animals/COW-A7F8B2
```

### `GET /api/animals/by-owner/{iin}`
```bash
curl http://localhost:8000/api/animals/by-owner/880101300123
```

### `DELETE /api/animals/{animal_id}` → `204`

### `GET /api/stats`
```json
{
  "total": 17,
  "per_species": [
    {"species": "cattle", "count": 9},
    {"species": "sheep",  "count": 6},
    {"species": "horse",  "count": 2}
  ],
  "recent": [
    {"id":"COW-A7F8B2","species":"cattle","owner_iin":"880101300123","registered_at":"..."}
  ]
}
```
