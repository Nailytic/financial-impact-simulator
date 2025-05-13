import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# Importar m車dulos del proyecto
from src.models.financial_model import FinancialModel
from src.visualization.dashboard import FinancialDashboard
from src.utils.data_processor import DataProcessor

# Configuraci車n de la p芍gina
st.set_page_config(
    page_title="Simulador de Impacto Financiero",
    page_icon="??",
    layout="wide"
)

# T赤tulo y descripci車n
st.title("Simulador Predictivo de Impacto Financiero")
st.markdown("""
    Esta aplicaci車n permite simular el impacto financiero de diferentes decisiones estrat谷gicas
    utilizando modelos predictivos y an芍lisis de escenarios.
""")

# Funci車n para cargar datos de ejemplo
@st.cache_data
def load_sample_data():
    # Si existe un archivo de datos, cargarlo
    if os.path.exists("data/raw/financial_sample.csv"):
        return pd.read_csv("data/raw/financial_sample.csv")
    
    # Caso contrario, generar datos sint谷ticos
    start_date = datetime(2018, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generar valor base con tendencia y estacionalidad
    np.random.seed(42)
    n = len(date_range)
    trend = np.linspace(1000, 2000, n)  # Tendencia al alza
    
    # Estacionalidad anual
    annual_seasonality = 200 * np.sin(2 * np.pi * np.arange(n) / 365)
    
    # Estacionalidad semanal
    weekly_seasonality = 50 * np.sin(2 * np.pi * np.arange(n) / 7)
    
    # Ruido
    noise = np.random.normal(0, 50, n)
    
    # Valor final
    y = trend + annual_seasonality + weekly_seasonality + noise
    
    # Factores externos
    marketing_spend = np.random.normal(100, 10, n)
    interest_rate = 2 + 0.5 * np.sin(2 * np.pi * np.arange(n) / 365)
    market_index = 1000 + 100 * np.sin(2 * np.pi * np.arange(n) / 180) + np.cumsum(np.random.normal(0, 5, n))
    
    # Crear DataFrame
    df = pd.DataFrame({
        'ds': date_range,
        'y': y,
        'marketing_spend': marketing_spend,
        'interest_rate': interest_rate,
        'market_index': market_index
    })
    
    # Guardar datos de ejemplo
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/financial_sample.csv", index=False)
    
    return df

# Sidebar - Carga de datos
st.sidebar.header("Configuraci車n")

# Opci車n para cargar datos propios o usar ejemplo
data_option = st.sidebar.radio(
    "Datos para an芍lisis:",
    ["Usar datos de ejemplo", "Cargar mis propios datos"]
)

if data_option == "Cargar mis propios datos":
    uploaded_file = st.sidebar.file_uploader("Cargar archivo CSV", type=["csv"])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Seleccionar columnas para an芍lisis
            date_col = st.sidebar.selectbox("Selecciona columna de fecha:", df.columns)
            target_col = st.sidebar.selectbox("Selecciona columna objetivo:", df.columns)
            
            # Convertir a formato esperado
            df = df.rename(columns={date_col: 'ds', target_col: 'y'})
            df['ds'] = pd.to_datetime(df['ds'])
            
            # Seleccionar factores externos
            external_factors = st.sidebar.multiselect(
                "Selecciona factores externos:",
                [col for col in df.columns if col not in ['ds', 'y']]
            )
        except Exception as e:
            st.error(f"Error al cargar datos: {e}")
            df = load_sample_data()
            external_factors = ['marketing_spend', 'interest_rate', 'market_index']
    else:
        df = load_sample_data()
        external_factors = ['marketing_spend', 'interest_rate', 'market_index']
else:
    df = load_sample_data()
    external_factors = ['marketing_spend', 'interest_rate', 'market_index']

# Mostrar datos cargados
st.subheader("Datos financieros")
st.dataframe(df.head())

# Sidebar - Par芍metros de modelado
st.sidebar.header("Par芍metros del Modelo")

seasonality_mode = st.sidebar.selectbox(
    "Modo de estacionalidad:",
    ["multiplicative", "additive"],
    index=0
)

prediction_periods = st.sidebar.slider(
    "Per赤odos de predicci車n:",
    min_value=30,
    max_value=730,
    value=365,
    step=30
)

changepoint_prior_scale = st.sidebar.slider(
    "Escala de prior para puntos de cambio:",
    min_value=0.001,
    max_value=0.5,
    value=0.05,
    step=0.001,
    format="%.3f"
)

# Sidebar - Escenarios
st.sidebar.header("Configuraci車n de Escenarios")

# Escenarios predefinidos y personalizados
scenario_type = st.sidebar.radio(
    "Tipo de escenarios:",
    ["Predefinidos", "Personalizado"]
)

if scenario_type == "Predefinidos":
    # Escenarios predefinidos para demostraci車n
    use_scenarios = {
        "optimista": {
            "marketing_spend": 1.2,  # Aumento del 20% en gasto de marketing
            "interest_rate": 0.9      # Reducci車n del 10% en tasas de inter谷s
        },
        "pesimista": {
            "marketing_spend": 0.8,  # Reducci車n del 20% en gasto de marketing
            "interest_rate": 1.1      # Aumento del 10% en tasas de inter谷s
        },
        "marketing_focus": {
            "marketing_spend": 1.5,  # Aumento del 50% en gasto de marketing
            "interest_rate": 1.0      # Sin cambios en tasas de inter谷s
        }
    }
else:
    # Interfaz para crear escenarios personalizados
    st.sidebar.subheader("Crear escenario personalizado")
    
    scenario_name = st.sidebar.text_input("Nombre del escenario:", "mi_escenario")
    
    scenario_adjustments = {}
    # Asegurarse de que external_factors est谷 definido si se carga archivo propio sin factores
    if 'external_factors' not in locals():
        external_factors = [col for col in df.columns if col not in ['ds', 'y']]
    
    for factor in external_factors:
        adjustment = st.sidebar.slider(
            f"Ajuste para {factor}:",
            min_value=0.5,
            max_value=1.5,
            value=1.0,
            step=0.05
        )
        scenario_adjustments[factor] = adjustment
    
    use_scenarios = {
        scenario_name: scenario_adjustments
    }

# Bot車n para ejecutar simulaci車n
run_simulation = st.sidebar.button("Ejecutar Simulaci車n")

if run_simulation:
    with st.spinner("Ejecutando simulaci車n financiera..."):
        # Crear y entrenar modelo
        model = FinancialModel(
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale
        )
        
        model.fit(df, external_factors=external_factors)
        
        # Preparar datos futuros
        # La funci車n make_future_dataframe de Prophet ya genera 'ds'
        # Debemos a?adir los factores externos manualmente para las fechas futuras
        future_df = model.model.make_future_dataframe(periods=prediction_periods)
        
        # A?adir factores externos a los datos futuros
        # Para el ejemplo, usaremos el promedio hist車rico como base para la proyecci車n futura
        # Puedes modificar esto para usar proyecciones m芍s sofisticadas si los factores externos
        # tienen tendencias o proyecciones conocidas.
        for factor in external_factors:
            if factor in df.columns:
                future_df[factor] = df[factor].mean()  # Valor promedio como punto de partida
        
        # Generar resultados de escenarios
        scenario_results = model.simulate_scenarios(future_df, use_scenarios)
        
        # Calcular m谷tricas de impacto
        impact_metrics = model.calculate_impact_metrics(scenario_results)
        
        # Generar datos para an芍lisis de sensibilidad
        base_factors = {factor: 1.0 for factor in external_factors}
        sensitivity_data = DataProcessor.calculate_sensitivity_matrix(
            scenario_results,
            base_factors,
            adjustment_range=(-0.2, 0.2), # Rango de ajuste para sensibilidad
            steps=5
        )
        
        # Crear visualizaciones
        dashboard = FinancialDashboard()
        
        # Mostrar dashboard
        dashboard_fig = dashboard.create_financial_dashboard(
            scenario_results,
            impact_metrics,
            sensitivity_data
        )
        
        st.plotly_chart(dashboard_fig, use_container_width=True)
        
        # Detalles de impacto financiero
        st.subheader("Detalles de Impacto Financiero")
        
        impact_table = impact_metrics.pivot(
            index='scenario',
            columns='metric',
            values=['absolute_difference', 'percentage_difference']
        )
        
        st.dataframe(impact_table)
        
        # Descargar resultados
        st.download_button(
            label="Descargar resultados",
            data=impact_table.to_csv().encode('utf-8'),
            file_name='financial_impact_results.csv',
            mime='text/csv',
        )
else:
    # Mostrar informaci車n cuando no se ha ejecutado la simulaci車n
    st.info("Configure los par芍metros y escenarios, luego haga clic en 'Ejecutar Simulaci車n'.")
    
    # Mostrar ejemplo visual de lo que se obtendr芍
    st.image("docs/images/example_dashboard.png", caption="Ejemplo de dashboard de simulaci車n")

# Informaci車n adicional
st.markdown("""
## Acerca del Simulador

Este simulador utiliza modelos de series temporales y an芍lisis de escenarios para predecir el impacto financiero
de diferentes decisiones estrat谷gicas. La metodolog赤a combina:

- **Modelado predictivo** con Prophet para capturar tendencias y estacionalidad
- **An芍lisis de escenarios** para simular diferentes decisiones empresariales
- **An芍lisis de sensibilidad** para identificar los factores m芍s influyentes

### C車mo interpretar los resultados

- La **Comparaci車n de Escenarios** muestra la proyecci車n financiera para cada escenario
- El **Impacto Porcentual** cuantifica la diferencia de cada escenario respecto al escenario base
- El **An芍lisis de Sensibilidad** indica qu谷 variables tienen mayor influencia en los resultados

### Limitaciones

Este simulador es una herramienta de apoyo a la decisi車n y no reemplaza el juicio experto.
Las predicciones est芍n sujetas a incertidumbre y deben interpretarse como tendencias
potenciales, no como valores exactos.
""")