import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataProcessor:
    """
    Clase para procesamiento de datos financieros.
    """
    
    @staticmethod
    def load_financial_data(filepath, date_col=None, target_col=None):
        """
        Carga datos financieros desde un archivo CSV.
        
        Args:
            filepath (str): Ruta al archivo CSV
            date_col (str): Nombre de la columna de fechas
            target_col (str): Nombre de la columna objetivo
            
        Returns:
            pd.DataFrame: DataFrame con los datos cargados y procesados
        """
        df = pd.read_csv(filepath)
        
        # Convertir columna de fecha si se especifica
        if date_col and date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
        
        return df
    
    @staticmethod
    def prepare_external_factors(df, external_factors, future_periods=365, frequency='D'):
        """
        Prepara factores externos para predicciones futuras.
        
        Args:
            df (pd.DataFrame): DataFrame con datos históricos
            external_factors (list): Lista de columnas con factores externos
            future_periods (int): Número de períodos futuros
            frequency (str): Frecuencia de los datos ('D' para diario, 'M' para mensual, etc.)
            
        Returns:
            pd.DataFrame: DataFrame con factores externos para predicciones futuras
        """
        if 'ds' not in df.columns:
            raise ValueError("El DataFrame debe tener una columna 'ds' con fechas")
            
        # Obtener última fecha de los datos históricos
        last_date = df['ds'].max()
        
        # Crear rango de fechas futuras
        if frequency == 'D':
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=future_periods,
                freq='D'
            )
        elif frequency == 'M':
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1), # Changed to start from the day after the last historical date
                periods=future_periods,
                freq='MS'  # Month Start
            )
        else:
            raise ValueError(f"Frecuencia '{frequency}' no soportada")
            
        # Crear DataFrame futuro
        future_df = pd.DataFrame({'ds': future_dates})
        
        # Proyectar factores externos (usando la media o el último valor)
        for factor in external_factors:
            if factor in df.columns:
                # Opción: Usar el último valor conocido (o podrías usar la media, o un modelo más complejo)
                last_value = df[factor].iloc[-1]
                future_df[factor] = last_value
                
                # Esta implementación podría mejorarse según la naturaleza de cada factor
                # Por ejemplo, para marketing_spend podrías querer proyectar un aumento
                # Para interest_rate, podrías tener una proyección externa
                
        return future_df
    
    @staticmethod
    def generate_sensitivity_scenarios(base_factors, adjustment_range=(-0.5, 0.5), steps=5):
        """
        Genera escenarios para análisis de sensibilidad.
        
        Args:
            base_factors (dict): Factores base como {factor: valor_base}
            adjustment_range (tuple): Rango de ajustes como (min, max)
            steps (int): Número de pasos entre min y max
            
        Returns:
            dict: Escenarios generados como {nombre_escenario: {factor: ajuste}}
        """
        scenarios = {}
        
        # Crear ajustes
        adjustments = np.linspace(adjustment_range[0], adjustment_range[1], steps)
        
        # Generar escenarios para cada factor y ajuste
        for factor, base_value in base_factors.items():
            for adj in adjustments:
                # Evitar el escenario sin cambios (adj = 0)
                if abs(adj) < 1e-10:
                    continue
                    
                scenario_name = f"{factor}_{adj:+.2f}"
                scenarios[scenario_name] = {factor: 1 + adj}
                
        return scenarios
    
    @staticmethod
    def calculate_sensitivity_matrix(scenario_results, base_factors, adjustment_range=(-0.5, 0.5), steps=5):
        """
        Calcula matriz de sensibilidad a partir de resultados de escenarios.
        
        Args:
            scenario_results (dict): Resultados de escenarios
            base_factors (dict): Factores base
            adjustment_range (tuple): Rango de ajustes
            steps (int): Número de pasos
            
        Returns:
            pd.DataFrame: DataFrame con datos de sensibilidad
        """
        base_result = scenario_results['base']['yhat'].sum()
        adjustments = np.linspace(adjustment_range[0], adjustment_range[1], steps)
        
        sensitivity_data = []
        
        for factor in base_factors:
            for adj in adjustments:
                # Evitar el escenario sin cambios (adj = 0)
                if abs(adj) < 1e-10:
                    continue
                
                scenario_name = f"{factor}_{adj:+.2f}"
                
                if scenario_name in scenario_results:
                    scenario_result = scenario_results[scenario_name]['yhat'].sum()
                    impact_pct = ((scenario_result / base_result) - 1) * 100
                    
                    sensitivity_data.append({
                        'factor': factor,
                        'adjustment': adj,
                        'impact': impact_pct
                    })
        
        return pd.DataFrame(sensitivity_data)