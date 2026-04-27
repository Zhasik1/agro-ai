"""Streamlit page — identify an animal."""

from __future__ import annotations

import streamlit as st

from frontend.lib.api_client import BackendError, identify

SPECIES_EMOJI = {"cattle": "🐂", "sheep": "🐑", "horse": "🐎"}

st.set_page_config(page_title="Тану — MalChain", page_icon="🔍")
st.title("🔍 Жануарды тану")
st.caption("Суретті жүктеңіз — жүйе оны тіркелгендермен салыстырады.")

photo = st.file_uploader(
    "📸 Жануардың суреті", type=["jpg", "jpeg", "png"], key="identify_photo"
)
if st.button("🔍 Тану", use_container_width=True, disabled=photo is None):
    try:
        data = identify(photo.getvalue())
    except BackendError as exc:
        if exc.status == 422:
            st.warning("⚠️ Жануар табылмады (cattle / sheep / horse).")
        else:
            st.error(f"❌ {exc.status}: {exc.detail}")
        st.stop()

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
