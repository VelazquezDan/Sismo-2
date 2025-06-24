# 🌋 SISMOS MX - Predicción y Monitoreo
# Código completo con todas las funcionalidades

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import geopandas as gpd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from fpdf import FPDF
import matplotlib.pyplot as plt
import warnings

# Configuración inicial
warnings.filterwarnings('ignore')
st.set_page_config(
    page_title="SISMOS MX | Predicción Avanzada",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# FUNCIONES PRINCIPALES
# --------------------------

@st.cache_data
def load_historical_data():
    """Carga datos históricos simulados con patrones estacionales"""
    dates = pd.date_range(start='2000-01-01', end=datetime.today(), freq='D')
    n_dates = len(dates)
    
    # Patrones simulados
    rng = np.random.RandomState(42)
    base_mag = rng.uniform(3.0, 5.0, n_dates)
    
    # Estacionalidad
    seasonal_effect = 0.5 * np.sin(2 * np.pi * dates.dayofyear / 365)
    
    # Tendencias por región
    regions = {
        'Pacífico': {'lat_range': (14.5, 22.5), 'lon_range': (-105.0, -95.0), 'factor': 1.2},
        'Norte': {'lat_range': (22.6, 32.5), 'lon_range': (-118.3, -105.1), 'factor': 0.8},
        'Sur': {'lat_range': (14.5, 18.5), 'lon_range': (-95.0, -86.7), 'factor': 1.5}
    }
    
    data = []
    for date, base, seasonal in zip(dates, base_mag, seasonal_effect):
        for region, params in regions.items():
            mag = base * params['factor'] + seasonal + rng.uniform(-0.3, 0.3)
            lat = rng.uniform(*params['lat_range'])
            lon = rng.uniform(*params['lon_range'])
            
            data.append({
                'Fecha': date,
                'Magnitud': max(3.0, min(8.0, mag)),
                'Latitud': lat,
                'Longitud': lon,
                'Profundidad': rng.uniform(5, 150),
                'Región': region,
                'Dia_Año': date.dayofyear,
                'Mes': date.month
            })
    
    return pd.DataFrame(data)

@st.cache_data
def train_prediction_model(df):
    """Entrena modelo de predicción con datos históricos"""
    features = ['Latitud', 'Longitud', 'Dia_Año', 'Mes', 'Profundidad']
    target = 'Magnitud'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluación
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    feature_imp = pd.Series(model.feature_importances_, index=features)
    
    return model, mae, feature_imp

@st.cache_data
def generate_predictions(model, days_to_predict=30):
    """Genera predicciones futuras"""
    last_date = datetime.today()
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict+1)]
    
    predictions = []
    regions = ['Pacífico', 'Norte', 'Sur']
    
    for date in future_dates:
        for region in regions:
            # Parámetros base por región
            if region == 'Pacífico':
                lat, lon = np.random.uniform(16.0, 20.0), np.random.uniform(-103.0, -97.0)
                depth = np.random.uniform(10, 100)
            elif region == 'Norte':
                lat, lon = np.random.uniform(24.0, 30.0), np.random.uniform(-110.0, -105.0)
                depth = np.random.uniform(5, 80)
            else:  # Sur
                lat, lon = np.random.uniform(15.0, 17.0), np.random.uniform(-94.0, -92.0)
                depth = np.random.uniform(20, 120)
            
            # Características para predicción
            features = {
                'Latitud': lat,
                'Longitud': lon,
                'Dia_Año': date.dayofyear,
                'Mes': date.month,
                'Profundidad': depth
            }
            
            # Predecir magnitud
            mag_pred = model.predict(pd.DataFrame([features]))[0]
            
            # Clasificar riesgo
            if mag_pred > 6.5:
                riesgo = 'Muy Alto'
                color = '#FF0000'
            elif mag_pred > 5.5:
                riesgo = 'Alto'
                color = '#FF6B00'
            elif mag_pred > 4.5:
                riesgo = 'Moderado'
                color = '#FFC100'
            else:
                riesgo = 'Bajo'
                color = '#00B050'
            
            predictions.append({
                'Fecha': date,
                'Latitud': lat,
                'Longitud': lon,
                'Magnitud_Predicha': mag_pred,
                'Profundidad_Predicha': depth,
                'Riesgo': riesgo,
                'Color': color,
                'Región': region
            })
    
    return pd.DataFrame(predictions)

# --------------------------
# INTERFAZ DE USUARIO
# --------------------------

def main():
    # Cargar datos
    df = load_historical_data()
    model, mae, feature_imp = train_prediction_model(df)
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/f/fc/SEGOB_Logo.svg", width=200)
        st.title("Configuración")
        
        days_to_predict = st.slider("Días a predecir", 7, 90, 30)
        min_magnitude = st.slider("Filtrar por magnitud mínima", 3.0, 6.0, 4.0)
        
        st.markdown("---")
        st.markdown(f"**Precisión del modelo:** MAE = {mae:.2f}")
        st.markdown("**Variables importantes:**")
        st.write(feature_imp.sort_values(ascending=False))
        
        if st.button("Actualizar datos"):
            st.cache_data.clear()
            st.rerun()
    
    # Título principal
    st.title("🌋 Sistema de Predicción Sísmica para México")
    st.markdown("""
    <style>
    .big-font { font-size:18px !important; }
    </style>
    <p class="big-font">Monitor avanzado que combina datos históricos con modelos predictivos para identificar zonas de riesgo</p>
    """, unsafe_allow_html=True)
    
    # Pestañas principales
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔮 Predicciones", "🗺️ Mapa Interactivo", "📈 Análisis Histórico"])
    
    with tab1:
        st.header("Indicadores Clave")
        
        # Métricas
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sismos históricos", len(df))
        col2.metric("Magnitud máxima registrada", f"{df['Magnitud'].max():.1f}")
        col3.metric("Región más activa", df['Región'].value_counts().idxmax())
        col4.metric("Precisión del modelo", f"{mae:.2f} MAE")
        
        # Gráfico de actividad reciente
        st.subheader("Actividad Reciente (Últimos 30 días)")
        recent_data = df[df['Fecha'] >= (datetime.today() - timedelta(days=30))]
        fig = px.histogram(recent_data, x='Fecha', y='Magnitud', color='Región',
                          nbins=30, hover_data=['Profundidad'])
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Predicciones de Riesgo Sísmico")
        
        if st.button("Generar Predicciones", type="primary"):
            with st.spinner("Calculando predicciones..."):
                predictions = generate_predictions(model, days_to_predict)
                
                # Mostrar resumen
                st.subheader(f"Resumen de Predicciones para {days_to_predict} días")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(predictions.sort_values('Magnitud_Predicha', ascending=False)[[
                        'Fecha', 'Región', 'Magnitud_Predicha', 'Riesgo'
                    ]].head(10))
                
                with col2:
                    risk_counts = predictions['Riesgo'].value_counts()
                    fig = px.pie(risk_counts, 
                                values=risk_counts.values, 
                                names=risk_counts.index,
                                color=risk_counts.index,
                                color_discrete_map={
                                    'Muy Alto': '#FF0000',
                                    'Alto': '#FF6B00',
                                    'Moderado': '#FFC100',
                                    'Bajo': '#00B050'
                                })
                    st.plotly_chart(fig, use_container_width=True)
                
                # Mapa de calor de riesgo
                st.subheader("Mapa de Calor de Riesgo")
                fig = px.density_mapbox(predictions, 
                                      lat='Latitud', 
                                      lon='Longitud', 
                                      z='Magnitud_Predicha',
                                      radius=15,
                                      center=dict(lat=23.6, lon=-102.5),
                                      zoom=5,
                                      mapbox_style="stamen-terrain",
                                      hover_name='Riesgo',
                                      hover_data=['Fecha', 'Región'])
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Mapa Interactivo de Sismos")
        
        # Filtros para el mapa
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Fecha inicial", 
                                     value=datetime.today() - timedelta(days=90))
        with col2:
            end_date = st.date_input("Fecha final", 
                                   value=datetime.today())
        
        # Filtrar datos
        filtered_df = df[(df['Fecha'].dt.date >= start_date) & 
                        (df['Fecha'].dt.date <= end_date) &
                        (df['Magnitud'] >= min_magnitude)]
        
        # Crear mapa
        fig = px.scatter_mapbox(filtered_df, 
                              lat="Latitud", 
                              lon="Longitud",
                              size="Magnitud",
                              color="Región",
                              hover_name="Región",
                              hover_data=["Fecha", "Magnitud", "Profundidad"],
                              zoom=5,
                              height=700,
                              center={"lat": 23.6, "lon": -102.5})
        
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Análisis Histórico")
        
        # Series temporales
        st.subheader("Tendencia Anual")
        df['Año'] = df['Fecha'].dt.year
        annual_trend = df.groupby('Año')['Magnitud'].mean().reset_index()
        fig = px.line(annual_trend, x='Año', y='Magnitud', 
                     title="Magnitud Promedio por Año")
        st.plotly_chart(fig, use_container_width=True)
        
        # Análisis por región
        st.subheader("Distribución por Región")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(df, x='Región', y='Magnitud', 
                        color='Región',
                        title="Distribución de Magnitudes por Región")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            region_counts = df['Región'].value_counts()
            fig = px.pie(region_counts, 
                        values=region_counts.values, 
                        names=region_counts.index,
                        title="Proporción de Sismos por Región")
            st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap de actividad
        st.subheader("Patrones Estacionales")
        df['Mes'] = df['Fecha'].dt.month
        heatmap_data = df.groupby(['Mes', 'Región'])['Magnitud'].mean().unstack()
        fig = px.imshow(heatmap_data.T,
                       labels=dict(x="Mes", y="Región", color="Magnitud"),
                       x=heatmap_data.index,
                       y=heatmap_data.columns,
                       aspect="auto")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
