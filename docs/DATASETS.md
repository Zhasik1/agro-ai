# 🗂 Datasets — MalChain

> **All entries below were enumerated from Section 12 of the project brief.** No URLs have been
> fabricated. Where the brief did not commit to a specific license or BibTeX entry, the field is
> tagged with `// TODO: confirm` so a contributor can verify it before publication.

This document accompanies [data/datasets.yaml](../data/datasets.yaml) and the
[scripts/download_datasets.py](../scripts/download_datasets.py) CLI.

## Privacy & ethics note

These datasets contain animal photographs only. They do **not** include personally-identifiable
human imagery, and they may not be used to identify, track, or profile individual people. Some
datasets (e.g. Roboflow Universe entries) carry per-dataset licenses — review and accept the
license shown on the dataset page before downloading. Consult `docs/DATASETS.md` again before any
public release to confirm that all bundled artefacts comply with their upstream terms.

## How to download

```bash
# List every entry the CLI knows about
python scripts/download_datasets.py --list

# Pull every cattle dataset that supports automated download
python scripts/download_datasets.py --download cattle

# Pull everything (datasets in `mode: manual` are skipped with instructions)
python scripts/download_datasets.py --download all
```

Required tools:

* `git` — for `mode: git` entries.
* `huggingface-cli` (or the `datasets` Python library) — for `mode: huggingface`.
* `ROBOFLOW_API_KEY` env var — for the Roboflow entries (set it locally; never commit it).

Downloaded data lands in `data/datasets/<species>/<name>/` and is git-ignored.

---

## Cattle

| Name | Source | License | Notes |
| --- | --- | --- | --- |
| Cattely cattle-face images | [github.com/aideep1400/Cattely-Cattle-Face-Images-Dataset](https://github.com/aideep1400/Cattely-Cattle-Face-Images-Dataset) | see repo | Clone with `git`. Inspect the upstream README for folder layout. |
| Cows frontal-face — Zenodo 10535934 | [zenodo.org/records/10535934](https://zenodo.org/records/10535934) | open (per Zenodo card) | Auto-download via direct HTTPS archive (`INDIVIDUAL SUBJECTS Data.zip`, ~13.9 GB) configured in `data/datasets.yaml`. |
| Cows-detection (Hugging Face) | [huggingface.co/datasets/UniqueData/cows-detection-dataset](https://huggingface.co/datasets/UniqueData/cows-detection-dataset) | see dataset card | Pulled via `huggingface_hub` snapshot download. |
| PMC10869238 — Cattle biometrics via muzzle | [ncbi.nlm.nih.gov/pmc/articles/PMC10869238](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10869238/) | CC-BY (per article) | Open-access article — supplementary materials accessed via the page. |

### BibTeX (cattle)

```bibtex
@misc{cattely2024,
  title  = {Cattely Cattle Face Images Dataset},
  author = {{aideep1400 contributors}},
  url    = {https://github.com/aideep1400/Cattely-Cattle-Face-Images-Dataset},
  note   = {GitHub repository}
  // TODO: confirm citation
}

@dataset{cows_frontal_face_zenodo_10535934,
  title     = {Cows frontal-face dataset},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.10535934},
  url       = {https://zenodo.org/records/10535934}
  // TODO: confirm citation (authors, year)
}

@misc{uniquedata_cows_detection,
  title  = {Cows Detection Dataset},
  author = {{UniqueData}},
  url    = {https://huggingface.co/datasets/UniqueData/cows-detection-dataset},
  note   = {Hugging Face dataset card}
  // TODO: confirm citation
}

@article{pmc10869238,
  title   = {Cattle biometrics through muzzle images},
  journal = {PMC},
  url     = {https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10869238/}
  // TODO: confirm citation (authors, year, full title)
}
```

---

## Sheep

| Name | Source | License | Notes |
| --- | --- | --- | --- |
| Sheepface-107 (5,350 images / 107 individuals) | [mdpi.com/2077-0472/13/9/1718](https://www.mdpi.com/2077-0472/13/9/1718) | see paper (MDPI Agriculture) | Manual: request access through the MDPI article's supplementary section. |
| Roboflow Universe — sheep face | [universe.roboflow.com/search/sheep%20face](https://universe.roboflow.com/search/sheep%20face) | per-dataset | Browse the search page; each result has its own license. Use a Roboflow API key (`$ROBOFLOW_API_KEY`). |

### BibTeX (sheep)

```bibtex
@article{sheepface107_2023,
  title   = {Sheepface-107: A dataset of 5,350 sheep face images for 107 individuals},
  journal = {Agriculture (MDPI)},
  volume  = {13},
  number  = {9},
  pages   = {1718},
  year    = {2023},
  url     = {https://www.mdpi.com/2077-0472/13/9/1718}
  // TODO: confirm citation (authors)
}
```

---

## Horse

| Name | Source | License | Notes |
| --- | --- | --- | --- |
| Roboflow — horse-face (`dlabequicare`) | [universe.roboflow.com/dlabequicare/horse-face-nfeoi](https://universe.roboflow.com/dlabequicare/horse-face-nfeoi) | CC-BY-4.0 (per page) | Horse Face Object Detection (~2,660 images). Roboflow API/key supported; use `$ROBOFLOW_API_KEY`. |
| Horse Individual Identification (Hugging Face) | [huggingface.co/datasets/Mobiusi/Horse-Individual-Identification-Dataset](https://huggingface.co/datasets/Mobiusi/Horse-Individual-Identification-Dataset) | see dataset card | Snapshot via `huggingface_hub`. |
| PFERD — Poses for Equine Research Dataset | [github.com/Celiali/PFERD](https://github.com/Celiali/PFERD) | see repo | Optional pose / 3D extension. Companion paper: [Scientific Data (Nature)](https://www.nature.com/articles/s41597-024-03312-1). |

### BibTeX (horse)

```bibtex
@misc{roboflow_horse_face_dlabequicare,
  title  = {Horse Face Dataset},
  author = {{dlabequicare}},
  url    = {https://universe.roboflow.com/dlabequicare/horse-face-nfeoi},
  note   = {Roboflow Universe}
  // TODO: confirm citation
}

@misc{mobiusi_horse_id,
  title  = {Horse Individual Identification Dataset},
  author = {{Mobiusi}},
  url    = {https://huggingface.co/datasets/Mobiusi/Horse-Individual-Identification-Dataset},
  note   = {Hugging Face dataset card}
  // TODO: confirm citation
}

@article{pferd2024,
  title   = {PFERD: Poses for Equine Research Dataset},
  journal = {Scientific Data},
  year    = {2024},
  url     = {https://www.nature.com/articles/s41597-024-03312-1}
  // TODO: confirm citation (authors, volume)
}
```

---

## Roadmap species (no datasets confirmed yet)

* **Goat** — pending dataset curation.
* **Camel** — pending dataset curation.
* **Pig** — pending dataset curation.

Do **not** add entries to `data/datasets.yaml` for these species until a verified URL,
license, and contact have been recorded here.
