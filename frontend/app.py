"""MalChain — Streamlit home page."""

from __future__ import annotations

import os

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
SPECIES_EMOJI = {"cattle": "🐂", "sheep": "🐑", "horse": "🐎"}

st.set_page_config(page_title="MalChain", page_icon="🐂", layout="wide")

st.title("🐂🐑🐎 MalChain")
st.subheader("Әр мал — паспорт. Әр паспорт — мүмкіндік.")
st.caption(
    "Қазақстан мал шаруашылығына арналған биометриялық сәйкестендіру платформасы — "
    "Multi-species livestock biometric identification platform."
)

st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(
        """
### 🎯 Не істейді?
- 🐂 **Түрін анықтау** — суреттен жануардың түрін табады (КРС / қой / жылқы)
- 📸 **Биометриялық тіркеу** — әр жануарға цифрлық паспорт жасайды
- 🔍 **Сәйкестендіру** — жаңа суретті тіркелгендермен салыстырады
- 🚫 **Қайталанулардан қорғау** — субсидия алаяқтығының алдын алады

### 🚀 Қалай бастау керек?
1. **🆕 Тіркеу** парағына өтіп, малыңыздың суретін жүктеңіз
2. **🔍 Тану** парағында суретті тексеріңіз
3. **📊 Статистика** парағында жалпы көрсеткіштерді көріңіз
        """
    )

with col2:
    st.markdown("### 📊 Live статистика")
    try:
        response = requests.get(f"{BACKEND_URL}/api/stats", timeout=5)
        response.raise_for_status()
        data = response.json()
        st.metric("Барлығы", data["total"])
        for entry in data["per_species"]:
            emoji = SPECIES_EMOJI.get(entry["species"], "🐾")
            st.metric(f"{emoji} {entry['species']}", entry["count"])
    except requests.RequestException as exc:
        st.warning(f"Backend қолжетімсіз / Backend offline: {exc}")

st.divider()
st.caption(
    "🏆 CyberShield Hackathon 2026 — Kozybayev University — "
    "[ku.edu.kz/hackathon](https://ku.edu.kz/hackathon)"
)
