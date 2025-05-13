import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class FinancialDashboard:
    """
    Clase para generar visualizaciones del simulador financiero.
    """
    
    def __init__(self, theme='plotly_white'):
        """
        Inicializa el dashboard financiero.
        
        Args:
            theme (str): Tema de Plotly para las visualizaciones
        """
        self.theme = theme
        
    def plot_forecast_comparison(self, scenario_results, title="Comparación de Escenarios"):
        """
        Crea un gráfico comparativo de diferentes escenarios.
        
        Args:
            scenario_results (dict): Diccionario con resultados de cada escenario
            title (str): Título del gráfico
            
        Returns:
            plotly.graph_objects.Figure: Figura de Plotly
        """
        fig = go.Figure()
        
        colors = px.colors.qualitative.D3
        
        for i, (scenario_name, forecast) in enumerate(scenario_results.items()):
            color = colors[i % len(colors)]
            
            # Añadir línea principal
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                mode='lines',
                name=f'{scenario_name} (predicción)',
                line=dict(color=color),
                hovertemplate='Fecha: %{x}<br>Valor: %{y:,.2f}'
            ))
            
            # Añadir intervalo de confianza
            fig.add_trace(go.Scatter(
                x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
                y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
                fill='toself',
                fillcolor=f'rgba{tuple(list(px.colors.hex_to_rgb(color)) + [0.2])}',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip',
                showlegend=False
            ))
        
        # Configurar diseño
        fig.update_layout(
            title=title,
            xaxis_title='Fecha',
            yaxis_title='Valor Financiero',
            hovermode='x unified',
            template=self.theme,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        
        return fig
    
    def plot_impact_metrics(self, impact_metrics, title="Impacto por Escenario"):
        """
        Crea un gráfico de barras para comparar métricas de impacto.
        
        Args:
            impact_metrics (pd.DataFrame): DataFrame con métricas de impacto
            title (str): Título del gráfico
            
        Returns:
            plotly.graph_objects.Figure: Figura de Plotly
        """
        # Filtrar solo para la métrica 'yhat'
        df = impact_metrics[impact_metrics['metric'] == 'yhat'].copy()
        
        fig = px.bar(
            df,
            x='scenario',
            y='percentage_difference',
            color='scenario',
            text_auto='.2f',
            title=title,
            labels={
                'scenario': 'Escenario',
                'percentage_difference': 'Diferencia Porcentual (%)'
            },
            template=self.theme
        )
        
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        
        fig.update_layout(
            showlegend=False,
            yaxis=dict(tickformat='.2f')
        )
        
        return fig
    
    def plot_sensitivity_heatmap(self, sensitivity_data, title="Análisis de Sensibilidad"):
        """
        Crea un mapa de calor para análisis de sensibilidad.
        
        Args:
            sensitivity_data (pd.DataFrame): DataFrame con datos de sensibilidad
            title (str): Título del gráfico
            
        Returns:
            plotly.graph_objects.Figure: Figura de Plotly
        """
        # Pivotear datos para formato de matriz
        pivot_data = sensitivity_data.pivot(
            index='factor', 
            columns='adjustment', 
            values='impact'
        )
        
        fig = px.imshow(
            pivot_data,
            labels=dict(x="Ajuste", y="Factor", color="Impacto (%)"),
            x=pivot_data.columns,
            y=pivot_data.index,
            color_continuous_scale='RdBu_r',
            origin='lower',
            aspect='auto',
            title=title,
            template=self.theme
        )
        
        fig.update_layout(
            coloraxis_colorbar=dict(
                title="Impacto (%)",
                tickformat='.2f'
            )
        )
        
        # Añadir valores en cada celda
        for i, factor in enumerate(pivot_data.index):
            for j, adjustment in enumerate(pivot_data.columns):
                fig.add_annotation(
                    x=adjustment,
                    y=factor,
                    text=f"{pivot_data.loc[factor, adjustment]:.2f}%",
                    showarrow=False,
                    font=dict(color="black" if abs(pivot_data.loc[factor, adjustment]) < 10 else "white")
                )
        
        return fig
    
    def create_financial_dashboard(self, scenario_results, impact_metrics, sensitivity_data=None):
        """
        Crea un dashboard completo con múltiples visualizaciones.
        
        Args:
            scenario_results (dict): Resultados de escenarios
            impact_metrics (pd.DataFrame): Métricas de impacto
            sensitivity_data (pd.DataFrame): Datos de sensibilidad (opcional)
            
        Returns:
            plotly.graph_objects.Figure: Figura de Plotly con el dashboard
        """
        if sensitivity_data is not None:
            fig = make_subplots(
                rows=2, cols=2,
                specs=[[{"colspan": 2}, None],
                      [{}, {}]],
                subplot_titles=(
                    "Comparación de Escenarios Financieros",
                    "Impacto Porcentual por Escenario",
                    "Análisis de Sensibilidad"
                ),
                vertical_spacing=0.1,
                horizontal_spacing=0.05
            )
            
            # Añadir gráficos al dashboard
            comparison_fig = self.plot_forecast_comparison(scenario_results)
            for trace in comparison_fig.data:
                fig.add_trace(trace, row=1, col=1)
                
            impact_fig = self.plot_impact_metrics(impact_metrics)
            for trace in impact_fig.data:
                fig.add_trace(trace, row=2, col=1)
                
            sensitivity_fig = self.plot_sensitivity_heatmap(sensitivity_data)
            for trace in sensitivity_fig.data:
                fig.add_trace(trace, row=2, col=2)
        else:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=(
                    "Comparación de Escenarios Financieros",
                    "Impacto Porcentual por Escenario"
                ),
                vertical_spacing=0.1
            )
            
            # Añadir gráficos al dashboard
            comparison_fig = self.plot_forecast_comparison(scenario_results)
            for trace in comparison_fig.data:
                fig.add_trace(trace, row=1, col=1)
                
            impact_fig = self.plot_impact_metrics(impact_metrics)
            for trace in impact_fig.data:
                fig.add_trace(trace, row=2, col=1)
        
        # Configurar diseño global
        fig.update_layout(
            height=800,
            title_text="Dashboard de Simulación de Impacto Financiero",
            template=self.theme,
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        
        return fig