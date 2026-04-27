"""Streamlit page — registry statistics."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.lib.api_client import BackendError, stats as fetch_stats

SPECIES_EMOJI = {"cattle": "🐂", "sheep": "🐑", "horse": "🐎"}

st.set_page_config(page_title="Статистика — MalChain", page_icon="📊")
st.title("📊 Тіркеу статистикасы")

try:
    data = fetch_stats()
except BackendError as exc:
    st.error(f"Backend қолжетімсіз: {exc.detail}")
    st.stop()

cols = st.columns(len(data["per_species"]) + 1)
cols[0].metric("Барлығы", data["total"])
for col, entry in zip(cols[1:], data["per_species"]):
    emoji = SPECIES_EMOJI.get(entry["species"], "🐾")
    col.metric(f"{emoji} {entry['species']}", entry["count"])

st.divider()

st.subheader("Түрлер бойынша диаграмма")
df_counts = pd.DataFrame(data["per_species"])
if not df_counts.empty and df_counts["count"].sum() > 0:
    fig = px.bar(df_counts, x="species", y="count", color="species", text="count")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Әзірге деректер жоқ.")

st.subheader("Соңғы тіркеулер")
df_recent = pd.DataFrame(data["recent"])
if df_recent.empty:
    st.info("Тіркеулер жоқ.")
else:
    st.dataframe(df_recent, use_container_width=True)
