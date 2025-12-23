import streamlit as st
import plotly.io as pio
from pathlib import Path
import json

st.set_page_config(page_title="Vélos & Météo Paris", layout="wide")
st.title("🚲 Vélos & Météo — Visualisation")

fig_dir = Path("figures")  # même dossier que celui créé dans le notebook

fig1_path = fig_dir / "fig_velo_precipitation_time.json"
fig2_path = fig_dir / "fig_velo_pluie.json"
fig3_path = fig_dir / "fig_velo_temperature_time.json"
fig4_path = fig_dir / "fig_velo_temperature.json"

missing = [p for p in [fig1_path, fig2_path, fig3_path, fig4_path] if not p.exists()]
if missing:
    st.error("Fichiers manquants : " + ", ".join(str(p) for p in missing))
    st.stop()

fig1 = pio.read_json(fig1_path)
fig2 = pio.read_json(fig2_path)
fig3 = pio.read_json(fig3_path)
fig4 = pio.read_json(fig4_path)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Evolution du nombre de vélos dans le temps en fonction de la pluie")
    st.plotly_chart(fig1, use_container_width=True)
    
    st.subheader("Evolution du nombre de vélos dans le temps en fonction de la température")
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    st.subheader("🌧️ Vélos & pluie")
    st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("🌡️ Vélos vs température")
    st.plotly_chart(fig4, use_container_width=True)



col1, col2 = st.columns(2)
with col1:
    st.subheader("Evolution du nombre de vélos dans le temps en fonction de la pluie")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("🌧️ Vélos & pluie")
    st.plotly_chart(fig2, use_container_width=True)


