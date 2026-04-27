"""MalChain — Streamlit home page."""

from __future__ import annotations

import streamlit as st

from frontend.lib.api_client import BackendError, stats as fetch_stats

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
        data = fetch_stats()
        st.metric("Барлығы", data["total"])
        for entry in data["per_species"]:
            emoji = SPECIES_EMOJI.get(entry["species"], "🐾")
            st.metric(f"{emoji} {entry['species']}", entry["count"])
    except BackendError as exc:
        st.warning(f"Backend қолжетімсіз / Backend offline: {exc.detail}")

st.divider()
st.caption(
    "🏆 CyberShield Hackathon 2026 — Kozybayev University — "
    "[ku.edu.kz/hackathon](https://ku.edu.kz/hackathon)"
)
