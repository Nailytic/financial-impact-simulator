import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor # Aunque no se usa directamente en este ejemplo, es un import presente

class FinancialModel:
    """
    Modelo predictivo para simulación de impacto financiero de decisiones estratégicas.
    """
    
    def __init__(self, seasonality_mode='multiplicative', changepoint_prior_scale=0.05):
        """
        Inicializa el modelo financiero.
        
        Args:
            seasonality_mode: Modo de estacionalidad para Prophet ('multiplicative' o 'additive')
            changepoint_prior_scale: Escala de prior para puntos de cambio en Prophet
        """
        self.prophet_params = {
            'seasonality_mode': seasonality_mode,
            'changepoint_prior_scale': changepoint_prior_scale
        }
        self.model = None
        self.external_factors = []
        
    def fit(self, historical_data, date_col='ds', target_col='y', external_factors=None):
        """
        Entrena el modelo con datos históricos.
        
        Args:
            historical_data (pd.DataFrame): DataFrame con datos históricos
            date_col (str): Nombre de la columna de fechas
            target_col (str): Nombre de la columna objetivo (métricas financieras)
            external_factors (list): Lista de columnas con factores externos
        """
        # Preparar datos para Prophet
        df = historical_data.rename(columns={date_col: 'ds', target_col: 'y'})
        
        # Inicializar modelo
        model = Prophet(**self.prophet_params)
        
        # Añadir regresores externos si existen
        if external_factors:
            for factor in external_factors:
                if factor in df.columns:
                    model.add_regressor(factor)
                    self.external_factors.append(factor)
        
        # Entrenar modelo
        model.fit(df)
        self.model = model
        
        return self
    
    def predict(self, future_periods=365, future_df=None, scenario_adjustments=None):
        """
        Genera predicciones financieras.
        
        Args:
            future_periods (int): Número de períodos futuros para predecir
            future_df (pd.DataFrame): DataFrame con datos futuros (opcional)
            scenario_adjustments (dict): Ajustes de escenario como {factor: multiplicador}
            
        Returns:
            pd.DataFrame: DataFrame con predicciones
        """
        if self.model is None:
            raise ValueError("El modelo debe ser entrenado antes de predecir")
            
        # Crear dataframe futuro si no se proporciona
        if future_df is None:
            future_df = self.model.make_future_dataframe(periods=future_periods)
        
        # Aplicar ajustes de escenario si se proporcionan
        if scenario_adjustments and self.external_factors:
            for factor, adjustment in scenario_adjustments.items():
                if factor in self.external_factors and factor in future_df.columns:
                    # Aplicar ajuste solo a fechas futuras
                    last_historical_date = self.model.history['ds'].max()
                    mask = future_df['ds'] > last_historical_date
                    
                    future_df.loc[mask, factor] = future_df.loc[mask, factor] * adjustment
        
        # Realizar predicción
        forecast = self.model.predict(future_df)
        
        return forecast
    
    def simulate_scenarios(self, base_future_df, scenarios):
        """
        Simula múltiples escenarios financieros.
        
        Args:
            base_future_df (pd.DataFrame): DataFrame base para predicciones
            scenarios (dict): Diccionario de escenarios como {nombre_escenario: {factor: ajuste}}
            
        Returns:
            dict: Resultados de cada escenario como {nombre_escenario: forecast_df}
        """
        results = {}
        
        # Escenario base (sin ajustes)
        results['base'] = self.predict(future_df=base_future_df)
        
        # Simular cada escenario proporcionado
        for scenario_name, adjustments in scenarios.items():
            scenario_forecast = self.predict(
                future_df=base_future_df.copy(),
                scenario_adjustments=adjustments
            )
            results[scenario_name] = scenario_forecast
            
        return results
    
    def calculate_impact_metrics(self, scenario_results, metric_cols=None):
        """
        Calcula métricas de impacto para diferentes escenarios.
        
        Args:
            scenario_results (dict): Resultados de escenarios
            metric_cols (list): Columnas a incluir en las métricas
            
        Returns:
            pd.DataFrame: DataFrame con métricas comparativas
        """
        if not metric_cols:
            metric_cols = ['yhat', 'yhat_lower', 'yhat_upper']
            
        base_scenario = scenario_results['base']
        
        metrics = []
        
        for scenario_name, forecast in scenario_results.items():
            if scenario_name == 'base':
                continue
                
            # Calcular diferencias con el escenario base
            for col in metric_cols:
                absolute_diff = forecast[col].sum() - base_scenario[col].sum()
                percentage_diff = ((forecast[col].sum() / base_scenario[col].sum()) - 1) * 100
                
                metrics.append({
                    'scenario': scenario_name,
                    'metric': col,
                    'absolute_difference': absolute_diff,
                    'percentage_difference': percentage_diff
                })
                
        return pd.DataFrame(metrics)