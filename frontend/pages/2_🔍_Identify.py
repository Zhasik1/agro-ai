"""Streamlit page — identify an animal."""

from __future__ import annotations

import os

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
SPECIES_EMOJI = {"cattle": "🐂", "sheep": "🐑", "horse": "🐎"}

st.set_page_config(page_title="Тану — MalChain", page_icon="🔍")
st.title("🔍 Жануарды тану")
st.caption("Суретті жүктеңіз — жүйе оны тіркелгендермен салыстырады.")

photo = st.file_uploader(
    "📸 Жануардың суреті", type=["jpg", "jpeg", "png"], key="identify_photo"
)
if st.button("🔍 Тану", use_container_width=True, disabled=photo is None):
    files = {"photo": (photo.name, photo.getvalue(), photo.type)}
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/animals/identify", files=files, timeout=120
        )
    except requests.RequestException as exc:
        st.error(f"Backend қатесі: {exc}")
        st.stop()

    if response.status_code == 422:
        st.warning("⚠️ Жануар табылмады (cattle / sheep / horse).")
    elif response.status_code >= 400:
        st.error(f"❌ {response.status_code}: {response.text}")
    else:
        data = response.json()
        status = data.get("status")
        species = data.get("species")
        emoji = SPECIES_EMOJI.get(species or "", "🐾")
        st.subheader(f"{emoji} Түрі: {species}  •  Сенімділік: {data.get('detection_confidence', 0):.2f}")

        if status == "matched":
            best = data["best_match"]
            st.success(
                f"✅ **Сәйкестік табылды** — {best['animal_id']} "
                f"(ұқсастық {best['similarity']:.2%})"
            )
        elif status == "suspect":
            st.warning("⚠️ **Күмәнді сәйкестік** — алғашқы 3 кандидат:")
            for cand in data.get("candidates", [])[:3]:
                st.write(f"• {cand['animal_id']} — {cand['similarity']:.2%}")
        elif status == "new":
            st.info("🆕 **Жаңа жануар** — тіркеуге дайын. «🆕 Register» парағына өтіңіз.")
        else:
            st.error(data.get("message") or "Қате")

        st.json(data)
