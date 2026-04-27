"""Streamlit page — register a new animal."""

from __future__ import annotations

import os

import requests
import streamlit as st
from PIL import Image, ImageDraw

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
SPECIES_EMOJI = {"cattle": "🐂", "sheep": "🐑", "horse": "🐎"}

st.set_page_config(page_title="Тіркеу — MalChain", page_icon="🆕")
st.title("🆕 Жаңа малды тіркеу")
st.caption("Жануардың суретін жүктеп, иесі мен қосымша мәліметтерді енгізіңіз.")

with st.form("register_form", clear_on_submit=False):
    photo = st.file_uploader(
        "📸 Жануардың суреті (jpg / png, ≤ 10 MB)",
        type=["jpg", "jpeg", "png"],
    )
    col1, col2 = st.columns(2)
    with col1:
        owner_iin = st.text_input("👤 Иесінің ЖСН (12 сан)", max_chars=12)
        breed = st.text_input("🧬 Тұқымы (міндетті емес)")
    with col2:
        age = st.number_input("📅 Жасы (жыл)", min_value=0, max_value=40, value=0)
        weight = st.number_input("⚖️ Салмағы (кг)", min_value=0.0, max_value=2000.0, value=0.0)
    notes = st.text_area("📝 Ескертулер", max_chars=500)
    submitted = st.form_submit_button("✅ Тіркеу", use_container_width=True)

if submitted:
    if not photo:
        st.error("Сурет жүктелмеген.")
    elif not owner_iin or len(owner_iin) != 12 or not owner_iin.isdigit():
        st.error("ЖСН 12 саннан тұруы керек.")
    else:
        files = {"photo": (photo.name, photo.getvalue(), photo.type)}
        form = {"owner_iin": owner_iin}
        if age:
            form["age_years"] = str(int(age))
        if weight:
            form["weight_kg"] = str(weight)
        if breed:
            form["breed"] = breed
        if notes:
            form["notes"] = notes
        try:
            response = requests.post(
                f"{BACKEND_URL}/api/animals/register",
                files=files,
                data=form,
                timeout=120,
            )
        except requests.RequestException as exc:
            st.error(f"Backend қатесі: {exc}")
        else:
            if response.status_code == 201:
                payload = response.json()
                animal = payload["animal"]
                emoji = SPECIES_EMOJI.get(animal["species"], "🐾")
                st.success(
                    f"✅ Тіркелді: **{emoji} {animal['id']}** — "
                    f"{animal['species']} (қ��еткіш {payload['detection_confidence']:.2f})"
                )
                st.json(animal)
            elif response.status_code == 409:
                err = response.json()
                st.warning(
                    f"⚠️ Бұл жануар тіркелген: **{err.get('existing_id')}** "
                    f"(ұқсастығы {err.get('similarity', 0):.2%})"
                )
            elif response.status_code == 422:
                st.error(f"⚠️ Валидация: {response.json()}")
            else:
                st.error(f"❌ {response.status_code}: {response.text}")
