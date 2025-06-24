import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import geopandas as gpd
from sklearn.cluster import DBSCAN
from fpdf import FPDF
import base64
from PIL import Image
import io

# --------------------------
# CONFIGURACIÓN DE LA PÁGINA
# --------------------------
st.set_page_config(
    page_title="SISMOS MX | Monitoreo Avanzado",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")  # Archivo CSS que crearemos después

# --------------------------
# DATOS Y MODELOS
# --------------------------
@st.cache_data
def load_data():
    # Datos simulados - en producción usarías fuentes reales
    data = pd.DataFrame({
        'Fecha': pd.date_range(end=datetime.today(), periods=365),
        'Magnitud': np.random.uniform(3.0, 7.5, 365),
        'Latitud': np.random.uniform(14.5, 32.5, 365),
        'Longitud': np.random.uniform(-118.3, -86.7, 365),
        'Profundidad': np.random.uniform(5, 150, 365),
        'Municipio': np.random.choice(['Acapulco', 'CDMX', 'Oaxaca', 'Chiapas', 'Guerrero'], 365)
    })
    return data

@st.cache_data
def load_geodata():
    # GeoJSON simplificado de municipios
    return gpd.read_file("municipios_mx.geojson")

# --------------------------
# COMPONENTES DE LA UI
# --------------------------
def main():
    # Título con estilo
    st.markdown("""
    <div class="header">
        <h1>🌋 SISMOS MX | Monitoreo Avanzado</h1>
        <p>Sistema integral de análisis sísmico para México</p>
    </div>
    """, unsafe_allow_html=True)

    # Carga de datos
    df = load_data()
    municipios = load_geodata()

    # --------------------------
    # SIDEBAR CON CONTROLES
    # --------------------------
    with st.sidebar:
        st.image("logo.png", width=200)  # Tu logo aquí
        st.markdown("## Parámetros de Análisis")
        
        fecha_inicio = st.date_input("Fecha inicial", 
                                   value=datetime.today()-timedelta(days=30))
        fecha_fin = st.date_input("Fecha final", 
                                value=datetime.today())
        
        min_magnitud = st.slider("Magnitud mínima", 3.0, 7.0, 4.0)
        
        st.markdown("---")
        st.markdown("## Sistema de Alertas")
        alert_magnitud = st.slider("Magnitud para alerta", 5.0, 8.0, 6.0)
        alert_municipio = st.selectbox("Municipio para monitoreo", 
                                     municipios['NOMGEO'].unique())
        
        st.markdown("---")
        if st.button("Generar Reporte PDF"):
            generate_report(df, municipios)

    # --------------------------
    # DASHBOARD DE INDICADORES
    # --------------------------
    st.markdown("## 📊 Indicadores Clave")
    
    # Filtrado de datos por fechas seleccionadas
    df_filtrado = df[(df['Fecha'].dt.date >= fecha_inicio) & 
                    (df['Fecha'].dt.date <= fecha_fin) &
                    (df['Magnitud'] >= min_magnitud)]
    
    # Métricas
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sismos registrados", len(df_filtrado))
    with col2:
        st.metric("Magnitud máxima", f"{df_filtrado['Magnitud'].max():.1f}")
    with col3:
        st.metric("Municipio más activo", 
                 df_filtrado['Municipio'].value_counts().idxmax())
    with col4:
        st.metric("Profundidad promedio", 
                 f"{df_filtrado['Profundidad'].mean():.1f} km")

    # --------------------------
    # MAPA INTERACTIVO
    # --------------------------
    st.markdown("## 🗺️ Mapa de Peligro Sísmico y Predicción")
    
    # Análisis de clusters
    coords = df_filtrado[['Latitud', 'Longitud']].values
    clustering = DBSCAN(eps=0.5, min_samples=3).fit(coords)
    df_filtrado['cluster'] = clustering.labels_
    
    # Capas de peligro sísmico (datos simulados)
    capas_peligro = {
        'Alto': {'coords': [[-99.1, 19.4], [-98.2, 18.5], [-100.3, 17.8]], 'color': 'red'},
        'Medio': {'coords': [[-102.5, 22.1], [-101.3, 20.8], [-103.7, 19.5]], 'color': 'orange'},
        'Bajo': {'coords': [[-106.3, 28.6], [-107.8, 25.3], [-109.1, 23.4]], 'color': 'yellow'}
    }
    
    # Creación del mapa
    fig = px.scatter_mapbox(df_filtrado, 
                          lat="Latitud", 
                          lon="Longitud",
                          size="Magnitud",
                          color="cluster",
                          hover_name="Municipio",
                          hover_data=["Fecha", "Magnitud"],
                          zoom=5,
                          center={"lat": 23.6, "lon": -102.5})
    
    # Añadir capas de peligro
    for nivel, data in capas_peligro.items():
        fig.add_trace(go.Scattermapbox(
            lon = [c[0] for c in data['coords']],
            lat = [c[1] for c in data['coords']],
            mode = 'lines',
            fill = 'toself',
            name = f'Peligro {nivel}',
            line = dict(color=data['color'], width=2),
            opacity = 0.3
        ))
    
    fig.update_layout(
        mapbox_style="open-street-map",
        height=600,
        margin={"r":0,"t":0,"l":0,"b":0},
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # --------------------------
    # ANÁLISIS POR MUNICIPIO
    # --------------------------
    st.markdown("## 🏙️ Análisis de Riesgo por Municipio")
    
    municipio_seleccionado = st.selectbox(
        "Selecciona un municipio:", 
        options=municipios['NOMGEO'].unique()
    )
    
    # Geométrica del municipio seleccionado
    muni_geom = municipios[municipios['NOMGEO'] == municipio_seleccionado].geometry.iloc[0]
    
    # Filtrar sismos en el municipio
    gdf = gpd.GeoDataFrame(
        df_filtrado, 
        geometry=gpd.points_from_xy(df_filtrado.Longitud, df_filtrado.Latitud)
    )
    sismos_muni = gdf[gdf.intersects(muni_geom)]
    
    if not sismos_muni.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"### Estadísticas para {municipio_seleccionado}")
            st.metric("Sismos registrados", len(sismos_muni))
            st.metric("Magnitud máxima", f"{sismos_muni['Magnitud'].max():.1f}")
            st.metric("Frecuencia mensual", 
                     f"{len(sismos_muni)/30:.1f} sismos/día")
        
        with col2:
            fig = px.histogram(sismos_muni, x="Magnitud", nbins=10,
                              title=f"Distribución de magnitudes en {municipio_seleccionado}")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No se registraron sismos en {municipio_seleccionado} en el período seleccionado")

    # --------------------------
    # SISTEMA DE ALERTAS
    # --------------------------
    st.markdown("## 🔔 Sistema de Alertas Tempranas")
    
    # Verificar condiciones de alerta
    alerta_condicion = (df_filtrado['Magnitud'] >= alert_magnitud) & \
                      (df_filtrado['Municipio'] == alert_municipio)
    
    if alerta_condicion.any():
        st.error(f"🚨 ALERTA: Se detectaron {alerta_condicion.sum()} sismos importantes en {alert_municipio}")
        st.dataframe(df_filtrado[alerta_condicion])
    else:
        st.success(f"✅ No se detectaron sismos de magnitud {alert_magnitud}+ en {alert_municipio}")

# --------------------------
# FUNCIÓN PARA REPORTE PDF
# --------------------------
def generate_report(df, municipios):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Encabezado
    pdf.cell(200, 10, txt="Reporte Sísmico de México", ln=1, align='C')
    pdf.cell(200, 10, txt=f"Generado el: {datetime.now().strftime('%Y-%m-%d')}", ln=2, align='C')
    
    # Métricas clave
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Indicadores Clave:", ln=3)
    pdf.set_font("Arial", size=12)
    
    metrics = [
        f"Total de sismos: {len(df)}",
        f"Magnitud máxima: {df['Magnitud'].max():.1f}",
        f"Municipio más activo: {df['Municipio'].value_counts().idxmax()}",
        f"Profundidad promedio: {df['Profundidad'].mean():.1f} km"
    ]
    
    for metric in metrics:
        pdf.cell(200, 10, txt=metric, ln=4)
    
    # Guardar PDF
    pdf_output = pdf.output(dest='S').encode('latin1')
    st.download_button(
        label="Descargar Reporte Completo",
        data=pdf_output,
        file_name="reporte_sismico_mx.pdf",
        mime="application/pdf"
    )

if __name__ == "__main__":
    main()
