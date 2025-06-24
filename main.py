import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import geopandas as gpd
import io

# Configuración de la página
st.set_page_config(page_title="Sismos México", page_icon="🇲🇽", layout="wide")

# Título de la aplicación
st.title("🇲🇽 Monitor Sísmico de México")
st.markdown("Aplicación para analizar la actividad sísmica en la República Mexicana")

# Cargar geometría de México (simplificada)
@st.cache_data
def load_mexico_geometry():
    # GeoJSON simplificado de México
    mexico_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "México"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-117.125, 32.535], [-86.733, 18.283], [-86.733, 14.533], [-92.229, 14.538], [-117.125, 32.535]]]
                }
            }
        ]
    }
    return gpd.GeoDataFrame.from_features(mexico_geojson)

# Obtener datos de sismos en México
@st.cache_data(ttl=3600)
def get_mexico_earthquakes(days=30, min_magnitude=4.0):
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)
    
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": start_time.strftime("%Y-%m-%d"),
        "endtime": end_time.strftime("%Y-%m-%d"),
        "minmagnitude": min_magnitude,
        "maxlatitude": 32.535,
        "minlatitude": 14.533,
        "maxlongitude": -86.733,
        "minlongitude": -117.125
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
    days = st.slider("Número de días a mostrar", 1, 365, 30)
    min_magnitude = st.slider("Magnitud mínima", 2.0, 8.0, 4.0, step=0.1)
    
    st.markdown("---")
    st.header("Cargar Datos Locales")
    uploaded_file = st.file_uploader("Subir dataset de sismos (CSV)", type="csv")
    
    st.markdown("---")
    st.markdown("**Fuentes de datos:**")
    st.markdown("- [USGS Earthquake Hazards Program](https://earthquake.usgs.gov)")
    st.markdown("- [SSN - UNAM](https://www.ssn.unam.mx)")

# Obtener datos de USGS
df_usgs = get_mexico_earthquakes(days, min_magnitude)

# Cargar datos locales si se proporcionan
df_local = None
if uploaded_file is not None:
    try:
        df_local = pd.read_csv(uploaded_file)
        # Convertir columnas de fecha si existen
        if 'Fecha' in df_local.columns:
            df_local['Fecha'] = pd.to_datetime(df_local['Fecha'])
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")

# Mostrar resumen estadístico
st.header("📊 Resumen Estadístico - México")

if not df_usgs.empty or df_local is not None:
    col1, col2, col3 = st.columns(3)
    
    if not df_usgs.empty:
        col1.metric("Sismos USGS", len(df_usgs))
        col2.metric("Magnitud máxima (USGS)", f"{df_usgs['Magnitud'].max():.1f}")
    
    if df_local is not None:
        col3.metric("Sismos locales", len(df_local))
        if 'Magnitud' in df_local.columns:
            col2.metric("Magnitud máxima (Local)", f"{df_local['Magnitud'].max():.1f}")

# Visualización de mapas
st.header("🌍 Mapa de Actividad Sísmica")

mexico_gdf = load_mexico_geometry()

fig = px.choropleth(mexico_gdf, geojson=mexico_gdf.geometry, 
                    locations=mexico_gdf.index, 
                    color_continuous_scale="Blues",
                    projection="mercator")
fig.update_geos(fitbounds="locations", visible=False)

# Añadir sismos USGS
if not df_usgs.empty:
    fig.add_trace(go.Scattergeo(
        lon = df_usgs['Longitud'],
        lat = df_usgs['Latitud'],
        text = df_usgs['Lugar'] + "<br>Magnitud: " + df_usgs['Magnitud'].astype(str),
        marker = dict(
            size = df_usgs['Magnitud']*2,
            color = df_usgs['Magnitud'],
            colorscale = 'Viridis',
            showscale = True,
            colorbar_title = 'Magnitud (USGS)'
        ),
        name = 'USGS',
        hoverinfo = 'text'
    ))

# Añadir sismos locales si existen
if df_local is not None and 'Latitud' in df_local.columns and 'Longitud' in df_local.columns:
    fig.add_trace(go.Scattergeo(
        lon = df_local['Longitud'],
        lat = df_local['Latitud'],
        text = df_local.get('Lugar', 'Sismo local') + "<br>" + 
               df_local.get('Magnitud', '').astype(str),
        marker = dict(
            size = df_local.get('Magnitud', 4)*2,
            color = 'red',
            symbol = 'x'
        ),
        name = 'Local',
        hoverinfo = 'text'
    ))

fig.update_layout(
    title_text = 'Sismos en México',
    geo = dict(
        scope = 'north america',
        landcolor = 'rgb(217, 217, 217)',
        center=dict(lon=-102, lat=23),
        projection_scale=2
    )
)

st.plotly_chart(fig, use_container_width=True)

# Comparación de datasets
if df_local is not None:
    st.header("🔍 Comparación de Datasets")
    
    # Seleccionar columnas para comparar
    cols_to_compare = st.multiselect(
        "Seleccionar columnas para comparar",
        options=[col for col in df_local.columns if col in df_usgs.columns],
        default=['Magnitud', 'Profundidad (km)'] if 'Magnitud' in df_local.columns else []
    )
    
    if cols_to_compare:
        tab1, tab2 = st.tabs(["Distribuciones", "Series Temporales"])
        
        with tab1:
            for col in cols_to_compare:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=df_usgs[col],
                    name='USGS',
                    opacity=0.75
                ))
                fig.add_trace(go.Histogram(
                    x=df_local[col],
                    name='Local',
                    opacity=0.75
                ))
                fig.update_layout(
                    barmode='overlay',
                    title_text=f'Distribución de {col}',
                    xaxis_title=col,
                    yaxis_title='Conteo'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            if 'Fecha' in df_local.columns and 'Fecha' in df_usgs.columns:
                for col in cols_to_compare:
                    fig = go.Figure()
                    
                    # Agrupar por fecha para USGS
                    usgs_daily = df_usgs.resample('D', on='Fecha')[col].mean().reset_index()
                    fig.add_trace(go.Scatter(
                        x=usgs_daily['Fecha'],
                        y=usgs_daily[col],
                        name='USGS',
                        line=dict(color='blue')
                    ))
                    
                    # Agrupar por fecha para local
                    local_daily = df_local.resample('D', on='Fecha')[col].mean().reset_index()
                    fig.add_trace(go.Scatter(
                        x=local_daily['Fecha'],
                        y=local_daily[col],
                        name='Local',
                        line=dict(color='red')
                    ))
                    
                    fig.update_layout(
                        title_text=f'Serie temporal de {col}',
                        xaxis_title='Fecha',
                        yaxis_title=col
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No se encontró columna 'Fecha' en ambos datasets para comparación temporal")

# Análisis por región
st.header("📍 Análisis por Región")

# Definir regiones sísmicas de México
regiones = {
    "Pacífico": {"lat": 18.0, "lon": -103.0, "radius": 3},
    "Golfo": {"lat": 20.0, "lon": -95.0, "radius": 3},
    "Norte": {"lat": 28.0, "lon": -105.0, "radius": 4},
    "Centro": {"lat": 19.4, "lon": -99.1, "radius": 2},
    "Sur": {"lat": 16.0, "lon": -95.0, "radius": 3}
}

selected_region = st.selectbox("Seleccionar región", options=list(regiones.keys()))

if not df_usgs.empty:
    region_data = regiones[selected_region]
    
    # Filtrar sismos en la región seleccionada
    df_region = df_usgs[
        (df_usgs['Latitud'] >= region_data['lat'] - region_data['radius']) &
        (df_usgs['Latitud'] <= region_data['lat'] + region_data['radius']) &
        (df_usgs['Longitud'] >= region_data['lon'] - region_data['radius']) &
        (df_usgs['Longitud'] <= region_data['lon'] + region_data['radius'])
    ]
    
    if not df_region.empty:
        st.subheader(f"Actividad sísmica en {selected_region}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total de sismos", len(df_region))
            st.metric("Magnitud máxima", f"{df_region['Magnitud'].max():.1f}")
        
        with col2:
            st.metric("Profundidad promedio", f"{df_region['Profundidad (km)'].mean():.1f} km")
            st.metric("Último sismo", df_region['Fecha'].max().strftime("%Y-%m-%d"))
        
        # Mapa de la región
        fig = px.scatter_mapbox(df_region, 
                               lat="Latitud", 
                               lon="Longitud", 
                               size="Magnitud",
                               color="Magnitud",
                               hover_name="Lugar",
                               hover_data=["Fecha", "Profundidad (km)"],
                               zoom=5,
                               center={"lat": region_data['lat'], "lon": region_data['lon']},
                               title=f"Sismos en {selected_region}")
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No se encontraron sismos en la región {selected_region} en el período seleccionado")
