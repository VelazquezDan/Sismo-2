import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import requests
from datetime import datetime, timedelta

# Configuración de la página
st.set_page_config(page_title="Monitor de Sismos", page_icon="🌍", layout="wide")

# Título de la aplicación
st.title("🌍 Monitor de Actividad Sísmica Mundial")
st.markdown("Esta aplicación muestra información sobre sismos recientes en todo el mundo, utilizando datos del Servicio Geológico de los Estados Unidos (USGS).")

# Obtener datos de sismos
@st.cache_data(ttl=3600)  # Cachear datos por 1 hora
def get_earthquake_data(days=30, min_magnitude=4.0):
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)
    
    url = f"https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": start_time.strftime("%Y-%m-%d"),
        "endtime": end_time.strftime("%Y-%m-%d"),
        "minmagnitude": min_magnitude,
        "orderby": "time"
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    earthquakes = []
    for feature in data["features"]:
        props = feature["properties"]
        geometry = feature["geometry"]
        earthquakes.append({
            "Fecha": pd.to_datetime(props["time"], unit="ms"),
            "Magnitud": props["mag"],
            "Lugar": props["place"],
            "Profundidad (km)": geometry["coordinates"][2],
            "Latitud": geometry["coordinates"][1],
            "Longitud": geometry["coordinates"][0],
            "Tipo": props.get("type", "earthquake"),
            "ID": props["code"]
        })
    
    return pd.DataFrame(earthquakes)

# Sidebar con controles
with st.sidebar:
    st.header("Configuración")
    days = st.slider("Número de días a mostrar", 1, 90, 30)
    min_magnitude = st.slider("Magnitud mínima", 2.0, 8.0, 4.0, step=0.1)
    
    st.markdown("---")
    st.markdown("**Datos proporcionados por:**")
    st.markdown("[USGS Earthquake Hazards Program](https://earthquake.usgs.gov)")
    st.markdown("---")
    st.markdown("Creado con Streamlit")

# Obtener datos
df = get_earthquake_data(days, min_magnitude)

# Mostrar resumen estadístico
st.header("📊 Resumen Estadístico")
if not df.empty:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de sismos", len(df))
    col2.metric("Magnitud máxima", f"{df['Magnitud'].max():.1f}")
    col3.metric("Magnitud promedio", f"{df['Magnitud'].mean():.1f}")
    col4.metric("Profundidad promedio", f"{df['Profundidad (km)'].mean():.1f} km")
    
    # Mostrar tabla con los sismos más fuertes
    st.subheader("🔝 Sismos más fuertes")
    st.dataframe(df.sort_values("Magnitud", ascending=False).head(10), 
                 column_config={
                     "Fecha": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss"),
                     "Magnitud": st.column_config.NumberColumn(format="%.1f"),
                     "Profundidad (km)": st.column_config.NumberColumn(format="%.1f km")
                 })
else:
    st.warning("No se encontraron sismos con los criterios seleccionados.")

# Gráficos y mapas
if not df.empty:
    st.header("📈 Visualizaciones")
    
    # Pestañas para diferentes visualizaciones
    tab1, tab2, tab3 = st.tabs(["Mapa de Sismos", "Distribución de Magnitudes", "Actividad Diaria"])
    
    with tab1:
        st.subheader("🌐 Mapa de Sismos")
        fig = px.scatter_geo(df, 
                            lat="Latitud", 
                            lon="Longitud", 
                            size="Magnitud",
                            color="Magnitud",
                            hover_name="Lugar",
                            hover_data=["Fecha", "Profundidad (km)"],
                            projection="natural earth",
                            title=f"Sismos de magnitud ≥ {min_magnitude} en los últimos {days} días",
                            color_continuous_scale="Viridis_r")
        fig.update_geos(showcoastlines=True, coastlinecolor="Black", showland=True, landcolor="lightgray")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📊 Distribución de Magnitudes")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df["Magnitud"], bins=20, edgecolor="black", color="skyblue")
        ax.set_xlabel("Magnitud")
        ax.set_ylabel("Número de sismos")
        ax.set_title(f"Distribución de magnitudes (≥ {min_magnitude})")
        st.pyplot(fig)
        
        # Gráfico de profundidad vs magnitud
        st.subheader("📉 Profundidad vs Magnitud")
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(df["Profundidad (km)"], df["Magnitud"], 
                            c=df["Magnitud"], cmap="viridis", alpha=0.6)
        ax.set_xlabel("Profundidad (km)")
        ax.set_ylabel("Magnitud")
        ax.set_title("Relación entre profundidad y magnitud")
        plt.colorbar(scatter, label="Magnitud")
        st.pyplot(fig)
    
    with tab3:
        st.subheader("📅 Actividad Sísmica Diaria")
        daily_counts = df.resample("D", on="Fecha").size()
        fig, ax = plt.subplots(figsize=(10, 6))
        daily_counts.plot(kind="bar", ax=ax, color="salmon", edgecolor="black")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Número de sismos")
        ax.set_title(f"Sismos diarios (≥ {min_magnitude})")
        plt.xticks(rotation=45)
        st.pyplot(fig)
else:
    st.warning("No hay datos para mostrar las visualizaciones.")
