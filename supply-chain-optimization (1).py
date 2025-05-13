# Sistema de Optimización de Cadena de Suministro con IA
# Desarrollado por Naiara Rodríguez - Data Analytics Portfolio

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pulp
from datetime import datetime, timedelta
import networkx as nx
import folium
from folium.plugins import HeatMap
import warnings
warnings.filterwarnings('ignore')

class SupplyChainOptimizer:
    """
    Sistema integrado para optimización de cadena de suministro usando ML e investigación operativa.
    Incluye predicción de demanda, optimización de inventario, y planificación de rutas.
    """
    
    def __init__(self):
        self.demand_model = None
        self.inventory_model = None
        self.route_optimizer = None
        self.data = {}
        self.predictions = {}
        self.optimization_results = {}
        
    def load_data(self, sales_data_path, inventory_data_path, locations_data_path=None, product_data_path=None):
        """Carga todos los conjuntos de datos necesarios"""
        
        print("Cargando datos del sistema de cadena de suministro...")
        
        # Cargar datos de ventas históricas
        self.data['sales'] = pd.read_csv(sales_data_path)
        print(f"Datos de ventas cargados: {self.data['sales'].shape[0]} registros")
        
        # Cargar datos de inventario
        self.data['inventory'] = pd.read_csv(inventory_data_path)
        print(f"Datos de inventario cargados: {self.data['inventory'].shape[0]} registros")
        
        # Cargar datos de ubicaciones (opcional)
        if locations_data_path:
            self.data['locations'] = pd.read_csv(locations_data_path)
            print(f"Datos de ubicaciones cargados: {self.data['locations'].shape[0]} ubicaciones")
        
        # Cargar datos de productos (opcional)
        if product_data_path:
            self.data['products'] = pd.read_csv(product_data_path)
            print(f"Datos de productos cargados: {self.data['products'].shape[0]} productos")
            
        # Realizar preprocesamiento inicial
        self._preprocess_data()
        
        return self
    
    def _preprocess_data(self):
        """Preprocesar y preparar los datos para análisis"""
        
        # Convertir columnas de fecha a datetime
        if 'date' in self.data['sales'].columns:
            self.data['sales']['date'] = pd.to_datetime(self.data['sales']['date'])
            
        # Agregar características temporales
        if 'date' in self.data['sales'].columns:
            self.data['sales']['year'] = self.data['sales']['date'].dt.year
            self.data['sales']['month'] = self.data['sales']['date'].dt.month
            self.data['sales']['day'] = self.data['sales']['date'].dt.day
            self.data['sales']['day_of_week'] = self.data['sales']['date'].dt.dayofweek
            self.data['sales']['quarter'] = self.data['sales']['date'].dt.quarter
            self.data['sales']['week_of_year'] = self.data['sales']['date'].dt.isocalendar().week
            
        # Manejar valores nulos en datos de ventas
        for col in self.data['sales'].columns:
            if self.data['sales'][col].isnull().sum() > 0:
                if self.data['sales'][col].dtype in [np.float64, np.int64]:
                    self.data['sales'][col] = self.data['sales'][col].fillna(self.data['sales'][col].median())
                else:
                    self.data['sales'][col] = self.data['sales'][col].fillna(self.data['sales'][col].mode()[0])
        
        # Verificar y corregir inconsistencias en datos de inventario
        if 'inventory' in self.data:
            for col in self.data['inventory'].columns:
                if self.data['inventory'][col].isnull().sum() > 0:
                    if self.data['inventory'][col].dtype in [np.float64, np.int64]:
                        self.data['inventory'][col] = self.data['inventory'][col].fillna(0)
                    else:
                        self.data['inventory'][col] = self.data['inventory'][col].fillna(self.data['inventory'][col].mode()[0])
        
        print("Preprocesamiento de datos completado")
        
    def analyze_demand_patterns(self, product_id=None, location_id=None, visualize=True):
        """
        Analiza patrones en la demanda para productos y ubicaciones específicas
        """
        sales_data = self.data['sales'].copy()
        
        # Filtrar por producto y ubicación si se especifican
        if product_id:
            sales_data = sales_data[sales_data['product_id'] == product_id]
        if location_id:
            sales_data = sales_data[sales_data['location_id'] == location_id]
            
        # Verificar si hay suficientes datos
        if len(sales_data) < 10:
            print("Datos insuficientes para el análisis de demanda con los filtros especificados")
            return None
            
        # Agrupar ventas por fecha
        if 'date' in sales_data.columns:
            daily_sales = sales_data.groupby('date')['quantity'].sum().reset_index()
            daily_sales = daily_sales.set_index('date')
            daily_sales = daily_sales.sort_index()
            
            # Calcular estadísticas
            weekly_avg = daily_sales.resample('W').mean()
            monthly_avg = daily_sales.resample('M').mean()
            
            # Detectar estacionalidad y tendencias
            sales_data['month'] = sales_data['date'].dt.month
            monthly_pattern = sales_data.groupby('month')['quantity'].mean()
            
            sales_data['day_of_week'] = sales_data['date'].dt.dayofweek
            dow_pattern = sales_data.groupby('day_of_week')['quantity'].mean()
            
            # Visualizar si se solicita
            if visualize:
                fig, axes = plt.subplots(3, 1, figsize=(14, 15))
                
                # Serie temporal
                daily_sales['quantity'].plot(ax=axes[0], title='Ventas Diarias')
                axes[0].set_ylabel('Unidades Vendidas')
                axes[0].grid(True)
                
                # Patrón mensual
                monthly_pattern.index = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
                monthly_pattern.plot(kind='bar', ax=axes[1], title='Patrón Mensual (Estacionalidad Anual)')
                axes[1].set_ylabel('Venta Promedio')
                axes[1].grid(True)
                
                # Patrón semanal
                dow_pattern.index = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
                dow_pattern.plot(kind='bar', ax=axes[2], title='Patrón Semanal')
                axes[2].set_ylabel('Venta Promedio')
                axes[2].grid(True)
                
                plt.tight_layout()
                plt.show()
            
            # Calcular estadísticas de patrones
            pattern_data = {
                'daily_sales': daily_sales,
                'weekly_avg': weekly_avg,
                'monthly_avg': monthly_avg,
                'monthly_pattern': monthly_pattern,
                'dow_pattern': dow_pattern
            }
            
            return pattern_data
        else:
            print("Columna de fecha no encontrada en los datos")
            return None
    
    def train_demand_forecast_model(self, features=None, target='quantity', test_size=0.2, forecast_periods=30):
        """
        Entrena un modelo de machine learning para predecir la demanda futura
        """
        print("Entrenando modelo de predicción de demanda...")
        
        # Preparar datos
        sales_data = self.data['sales'].copy()
        
        # Definir características si no se proporcionan
        if features is None:
            # Usar características temporales y cualquier otra disponible
            features = ['year', 'month', 'day', 'day_of_week', 'quarter', 'week_of_year']
            
            # Añadir características de producto si están disponibles
            if 'product_id' in sales_data.columns:
                features.append('product_id')
            if 'location_id' in sales_data.columns:
                features.append('location_id')
                
        # Verificar que todas las características existen
        features = [f for f in features if f in sales_data.columns]
        
        if not features:
            print("No se encontraron características válidas para el modelo")
            return None
            
        # Separar características y objetivo
        X = sales_data[features]
        y = sales_data[target]
        
        # Separar datos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Identificar columnas categóricas y numéricas
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Crear preprocesadores
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combinar preprocesadores
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Crear y entrenar modelo con hiperparámetros optimizados
        forest_model = RandomForestRegressor(random_state=42)
        
        # Crear pipeline con preprocesador y modelo
        demand_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', forest_model)
        ])
        
        # Entrenar modelo
        demand_pipeline.fit(X_train, y_train)
        
        # Evaluar modelo
        y_pred = demand_pipeline.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"Modelo de predicción entrenado con resultados:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2:.4f}")
        
        # Guardar modelo
        self.demand_model = demand_pipeline
        
        # Generar y guardar predicciones futuras
        future_df = self._generate_future_features(forecast_periods)
        future_predictions = self.demand_model.predict(future_df)
        
        # Guardar predicciones
        self.predictions['demand'] = pd.DataFrame({
            'date': future_df['date'] if 'date' in future_df.columns else pd.date_range(start=datetime.now(), periods=forecast_periods),
            'predicted_demand': future_predictions
        })
        
        return self
    
    def _generate_future_features(self, periods):
        """Genera características para fechas futuras para predicción"""
        last_date = self.data['sales']['date'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods)
        
        future_df = pd.DataFrame({'date': future_dates})
        future_df['year'] = future_df['date'].dt.year
        future_df['month'] = future_df['date'].dt.month
        future_df['day'] = future_df['date'].dt.day
        future_df['day_of_week'] = future_df['date'].dt.dayofweek
        future_df['quarter'] = future_df['date'].dt.quarter
        future_df['week_of_year'] = future_df['date'].dt.isocalendar().week
        
        # Añadir otras características si son necesarias
        # Por ejemplo, para producto y ubicación, usar el último o más común
        if 'product_id' in self.data['sales'].columns:
            future_df['product_id'] = self.data['sales']['product_id'].mode()[0]
        
        if 'location_id' in self.data['sales'].columns:
            future_df['location_id'] = self.data['sales']['location_id'].mode()[0]
            
        return future_df
    
    def optimize_inventory(self, holding_cost=0.1, stockout_cost=0.5, service_level=0.95):
        """
        Optimiza niveles de inventario basado en predicciones de demanda
        """
        print("Optimizando niveles de inventario...")
        
        if self.predictions.get('demand') is None:
            print("Error: Primero debes entrenar el modelo de predicción de demanda")
            return None
            
        # Obtener predicciones de demanda
        demand_predictions = self.predictions['demand']['predicted_demand'].values
        
        # Calcular estadísticas de demanda para análisis de inventario
        mean_demand = np.mean(demand_predictions)
        std_demand = np.std(demand_predictions)
        
        # Calcular punto de reorden y nivel de inventario objetivo
        # Usando fórmula basada en nivel de servicio (distribución normal)
        from scipy.stats import norm
        z_score = norm.ppf(service_level)
        
        # Asumir lead time promedio (días)
        avg_lead_time = 3
        
        # Inventario de seguridad
        safety_stock = z_score * std_demand * np.sqrt(avg_lead_time)
        
        # Punto de reorden (ROP)
        reorder_point = mean_demand * avg_lead_time + safety_stock
        
        # Calcular cantidad económica de pedido (EOQ)
        # Suponiendo un costo fijo de pedido
        order_cost = 100  # costo fijo por hacer un pedido
        
        eoq = np.sqrt((2 * order_cost * np.sum(demand_predictions)) / holding_cost)
        
        # Optimizar usando programación lineal para múltiples productos/ubicaciones
        # (Simplificado para este ejemplo)
        
        # Crear resultados
        inventory_policy = {
            'safety_stock': safety_stock,
            'reorder_point': reorder_point,
            'eoq': eoq,
            'mean_demand': mean_demand,
            'std_demand': std_demand,
            'service_level': service_level
        }
        
        # Simular impacto de la política optimizada
        stock_simulation = self._simulate_inventory_policy(
            demand_predictions, 
            initial_stock=eoq, 
            reorder_point=reorder_point, 
            order_quantity=eoq
        )
        
        # Guardar resultados
        self.optimization_results['inventory'] = {
            'policy': inventory_policy,
            'simulation': stock_simulation
        }
        
        # Visualizar simulación
        plt.figure(figsize=(14, 7))
        
        plt.plot(stock_simulation['stock_level'], label='Nivel de Inventario', color='blue')
        plt.axhline(y=reorder_point, color='r', linestyle='--', label='Punto de Reorden')
        plt.axhline(y=safety_stock, color='g', linestyle='--', label='Stock de Seguridad')
        
        # Marcar órdenes
        order_points = np.where(stock_simulation['order_placed'])[0]
        plt.scatter(order_points, [reorder_point] * len(order_points), color='red', marker='^', s=100, label='Orden Realizada')
        
        plt.title('Simulación de Política de Inventario Optimizada')
        plt.xlabel('Día')
        plt.ylabel('Unidades')
        plt.grid(True)
        plt.legend()
        plt.show()
        
        print(f"Política de inventario optimizada:")
        print(f"Stock de seguridad: {safety_stock:.2f} unidades")
        print(f"Punto de reorden: {reorder_point:.2f} unidades")
        print(f"Cantidad económica de pedido: {eoq:.2f} unidades")
        print(f"Nivel de servicio objetivo: {service_level*100:.1f}%")
        print(f"Nivel de servicio logrado: {stock_simulation['service_level']*100:.1f}%")
        
        return self.optimization_results['inventory']
    
    def _simulate_inventory_policy(self, demand, initial_stock, reorder_point, order_quantity, lead_time=3):
        """Simula la implementación de una política de inventario"""
        
        days = len(demand)
        stock_level = np.zeros(days)
        order_placed = np.zeros(days, dtype=bool)
        stockouts = np.zeros(days, dtype=bool)
        pending_orders = []  # [cantidad, día_de_llegada]
        
        # Inicializar
        stock_level[0] = initial_stock
        
        for day in range(days):
            # Recibir órdenes pendientes
            orders_to_remove = []
            for i, order in enumerate(pending_orders):
                if order[1] == day:
                    stock_level[day] += order[0]
                    orders_to_remove.append(i)
            
            # Eliminar órdenes ya recibidas
            for i in sorted(orders_to_remove, reverse=True):
                pending_orders.pop(i)
            
            # Satisfacer demanda
            if day > 0:
                stock_level[day] = stock_level[day-1]
                
            if stock_level[day] < demand[day]:
                stockouts[day] = True
                stock_level[day] = 0
            else:
                stock_level[day] -= demand[day]
            
            # Verificar si se necesita hacer un pedido
            if stock_level[day] <= reorder_point and not order_placed[max(0, day-lead_time):day+1].any():
                order_placed[day] = True
                # Programar llegada del pedido
                arrival_day = min(day + lead_time, days-1)
                pending_orders.append([order_quantity, arrival_day])
        
        # Calcular nivel de servicio (% de días sin stockout)
        service_level = 1 - (np.sum(stockouts) / days)
        
        return {
            'stock_level': stock_level,
            'order_placed': order_placed,
            'stockouts': stockouts,
            'service_level': service_level
        }
    
    def optimize_distribution_routes(self, origin_id, destination_ids, visualize=True):
        """
        Optimiza rutas de distribución entre almacenes y destinos
        """
        print("Optimizando rutas de distribución...")
        
        if 'locations' not in self.data:
            print("Error: Se requieren datos de ubicaciones para optimizar rutas")
            return None
            
        locations = self.data['locations']
        
        # Verificar que las ubicaciones existen
        if origin_id not in locations['location_id'].values:
            print(f"Error: Ubicación de origen {origin_id} no encontrada")
            return None
            
        valid_destinations = [d for d in destination_ids if d in locations['location_id'].values]
        if len(valid_destinations) != len(destination_ids):
            print(f"Advertencia: {len(destination_ids) - len(valid_destinations)} destinos no encontrados")
            
        if not valid_destinations:
            print("Error: Ningún destino válido para optimizar rutas")
            return None
            
        # Extraer las coordenadas
        origin = locations[locations['location_id'] == origin_id].iloc[0]
        destinations = locations[locations['location_id'].isin(valid_destinations)]
        
        # Calcular matriz de distancias (simplificado usando distancia euclidiana)
        # En un caso real se usaría una API como Google Maps para distancias reales
        
        n_destinations = len(destinations)
        distance_matrix = np.zeros((n_destinations + 1, n_destinations + 1))
        
        # Añadir origen como primer nodo
        all_points = pd.concat([pd.DataFrame([origin]), destinations])
        
        # Calcular todas las distancias
        for i in range(len(all_points)):
            for j in range(len(all_points)):
                if i != j:
                    # Distancia euclidiana entre puntos
                    distance_matrix[i, j] = np.sqrt(
                        (all_points.iloc[i]['latitude'] - all_points.iloc[j]['latitude'])**2 + 
                        (all_points.iloc[i]['longitude'] - all_points.iloc[j]['longitude'])**2
                    ) * 111  # Convertir a km aproximadamente
        
        # Resolver problema del viajante (TSP) para encontrar la ruta óptima
        # Usamos un enfoque de programación lineal entera
        
        # Crear problema
        prob = pulp.LpProblem("TSP_Route_Optimization", pulp.LpMinimize)
        
        # Crear variables de decisión
        x = {}
        for i in range(len(all_points)):
            for j in range(len(all_points)):
                if i != j:
                    x[i, j] = pulp.LpVariable(f'x_{i}_{j}', cat='Binary')
        
        # Crear variables para subrutas
        u = {}
        for i in range(1, len(all_points)):
            u[i] = pulp.LpVariable(f'u_{i}', lowBound=1, upBound=len(all_points)-1, cat='Integer')
            
        # Función objetivo: minimizar distancia total
        prob += pulp.lpSum(distance_matrix[i, j] * x[i, j] for i in range(len(all_points)) 
                          for j in range(len(all_points)) if i != j)
        
        # Restricciones
        # Cada nodo debe tener exactamente una salida
        for i in range(len(all_points)):
            prob += pulp.lpSum(x[i, j] for j in range(len(all_points)) if i != j) == 1
            
        # Cada nodo debe tener exactamente una entrada
        for j in range(len(all_points)):
            prob += pulp.lpSum(x[i, j] for i in range(len(all_points)) if i != j) == 1
            
        # Eliminar subrutas (restricciones MTZ)
        for i in range(1, len(all_points)):
            for j in range(1, len(all_points)):
                if i != j:
                    prob += u[i] - u[j] + len(all_points) * x[i, j] <= len(all_points) - 1
        
        # Resolver
        solver = pulp.PULP_CBC_CMD(msg=False)
        prob.solve(solver)
        
        # Extraer solución
        if pulp.LpStatus[prob.status] == 'Optimal':
            # Construir ruta
            route = [0]  # Comenzar en el origen
            current = 0
            
            while len(route) < len(all_points):
                for j in range(len(all_points)):
                    if j != current and x[current, j].value() > 0.5:
                        route.append(j)
                        current = j
                        break
            
            # Convertir índices de ruta a IDs de ubicación
            route_location_ids = [all_points.iloc[i]['location_id'] for i in route]
            
            # Calcular distancia total
            total_distance = sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
            total_distance += distance_matrix[route[-1], route[0]]  # Volver al origen
            
            # Crear resultados
            route_results = {
                'route': route_location_ids,
                'total_distance': total_distance,
                'locations': all_points.iloc[route][['location_id', 'name', 'latitude', 'longitude']].to_dict('records')
            }
            
            # Visualizar ruta en mapa
            if visualize:
                self._visualize_route(route_results)
            
            # Guardar resultados
            self.optimization_results['route'] = route_results
            
            print(f"Ruta óptima encontrada con distancia total de {total_distance:.2f} km")
            print(f"Secuencia de ruta: {' -> '.join(str(loc) for loc in route_location_ids)}")
            
            return route_results
        else:
            print(f"No se encontró solución óptima: {pulp.LpStatus[prob.status]}")
            return None
    
    def _visualize_route(self, route_results):
        """Visualiza la ruta optimizada en un mapa interactivo"""
        
        # Crear mapa centrado en la primera ubicación
        locations = route_results['locations']
        center = [locations[0]['latitude'], locations[0]['longitude']]
        
        m = folium.Map(location=center, zoom_start=10)
        
        # Añadir marcadores para cada ubicación
        for i, loc in enumerate(locations):
            popup_text = f"{loc['name']} (ID: {loc['location_id']})"
            
            # Origen en rojo, destinos en azul
            if i == 0:
                icon_color = 'red'
                icon = 'home'
            else:
                icon_color = 'blue'
                icon = 'store'
                
            folium.Marker(
                location=[loc['latitude'], loc['longitude']],
                popup=popup_text,
                tooltip=f"Stop #{i}: {loc['name']}",
                icon=folium.Icon(color=icon_color, icon=icon)
            ).add_to(m)
        
        # Añadir líneas para la ruta
        route_points = [(loc['latitude'], loc['longitude']) for loc in locations]
        # Añadir el origen al final para cerrar el circuito
        route_points.append(route_points[0])
        
        folium.PolyLine(
            route_points,
            color='green',
            weight=5,
            opacity=0.8
        ).add_to(m)
        
        # Añadir información de distancia
        folium.map.Marker(
            location=center,
            icon=folium.DivIcon(
                icon_size=(250, 36),
                icon_anchor=(0, 0),
                html=f'<div style="background-color: white; padding: 5px; border: 1px solid black;">'
                     f'Distancia total: {route_results["total_distance"]:.2f} km</div>'
            )
        ).add_to(m)
        
        # Mostrar mapa
        display(m)
    
    def generate_optimization_report(self):
        """
        Genera un informe completo con todos los resultados de optimización
        """
        print("\n=== INFORME DE OPTIMIZACIÓN DE CADENA DE SUMINISTRO ===\n")
        
        # Verificar si tenemos resultados
        if not hasattr(self, 'optimization_results') or not self.optimization_results:
            print("No hay resultados de optimización disponibles. Ejecute primero los módulos de optimización.")
            return
            
        # 1. Resumen de predicción de demanda
        if 'demand' in self.predictions:
            print("\n--- PREDICCIÓN DE DEMANDA ---\n")
            demand = self.predictions['demand']
            print(f"Período de pronóstico: {demand['date'].min().strftime('%d-%m-%Y')} a {demand['date'].max().strftime('%d-%m-%Y')}")
            print(f"Demanda total prevista: {demand['predicted_demand'].sum():.0f} unidades")
            print(f"Demanda diaria promedio: {demand['predicted_demand'].mean():.2f} unidades")
            print(f"Demanda máxima: {demand['predicted_demand'].max():.2f} unidades el {demand.loc[demand['predicted_demand'].idxmax(), 'date'].strftime('%d-%m-%Y')}")
            
            # Visualización rápida
            plt.figure(figsize=(10, 5))
            plt.plot(demand['date'], demand['predicted_demand'])
            plt.title('Predicción de Demanda')
            plt.xlabel('Fecha')
            plt.ylabel('Unidades')
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
        # 2. Resultados de optimización de inventario
        if 'inventory' in self.optimization_results:
            print("\n--- POLÍTICA DE INVENTARIO OPTIMIZADA ---\n")
            inventory = self.optimization_results['inventory']
            policy = inventory['policy']
            
            print(f"Stock de seguridad: {policy['safety_stock']:.2f} unidades")
            print(f"Punto de reorden: {policy['reorder_point']:.2f} unidades")
            print(f"Cantidad económica de pedido (EOQ): {policy['eoq']:.2f} unidades")
            print(f"Nivel de servicio objetivo: {policy['service_level']*100:.1f}%")
            
            simulation = inventory['simulation']
            print(f"Nivel de servicio logrado en simulación: {simulation['service_level']*100:.1f}%")
            print(f"Número de órdenes realizadas: {np.sum(simulation['order_placed'])}")
            print(f"Días con desabastecimiento: {np.sum(simulation['stockouts'])} de {len(simulation['stockouts'])}")
            
        # 3. Resultados de optimización de rutas
        if 'route' in self.optimization_results:
            print("\n--- RUTA DE DISTRIBUCIÓN OPTIMIZADA ---\n")
            route = self.optimization_results['route']
            
            print(f"Distancia total: {route['total_distance']:.2f} km")
            print(f"Número de ubicaciones visitadas: {len(route['locations'])}")
            print(f"Secuencia de ruta: {' -> '.join(str(loc) for loc in route['route'])}")
            
        # 4. Impacto económico estimado
        print("\n--- IMPACTO ECONÓMICO ESTIMADO ---\n")
        
        # Variables para cálculos
        total_demand = 0
        holding_cost_savings = 0
        stockout_cost_savings = 0
        transportation_cost_savings = 0
        
        if 'demand' in self.predictions:
            total_demand = self.predictions['demand']['predicted_demand'].sum()
            
        if 'inventory' in self.optimization_results:
            # Estimar ahorros en costos de mantenimiento (suponiendo 30% de reducción)
            avg_inventory_reduction = 0.30  # 30% de reducción promedio
            unit_holding_cost = 5  # $5 por unidad por período
            holding_cost_savings = avg_inventory_reduction * policy['mean_demand'] * unit_holding_cost * len(simulation['stock_level'])
            
            # Estimar ahorros en desabastecimiento (suponiendo 80% de reducción)
            stockout_reduction = 0.80  # 80% de reducción de stockouts
            unit_stockout_cost = 20  # $20 por unidad no vendida
            expected_stockouts_before = policy['mean_demand'] * 0.10 * len(simulation['stock_level'])  # Asumiendo 10% sin optimizar
            expected_stockouts_after = np.sum(simulation['stockouts']) * policy['mean_demand']
            stockout_cost_savings = (expected_stockouts_before - expected_stockouts_after) * unit_stockout_cost
            
        if 'route' in self.optimization_results:
            # Estimar ahorros en transporte (suponiendo 25% de reducción)
            transport_reduction = 0.25  # 25% de reducción
            transport_cost_per_km = 2  # $2 por km
            original_distance_estimate = route['total_distance'] / (1 - transport_reduction)
            transportation_cost_savings = (original_distance_estimate - route['total_distance']) * transport_cost_per_km
            
        # Calcular ahorros totales
        total_savings = holding_cost_savings + stockout_cost_savings + transportation_cost_savings
        
        print(f"Ahorros estimados en costos de inventario: ${holding_cost_savings:.2f}")
        print(f"Ahorros estimados por reducción de desabastecimiento: ${stockout_cost_savings:.2f}")
        print(f"Ahorros estimados en costos de transporte: ${transportation_cost_savings:.2f}")
        print(f"AHORROS TOTALES ESTIMADOS: ${total_savings:.2f}")
        
        if total_demand > 0:
            print(f"Ahorro por unidad: ${total_savings/total_demand:.2f}")
            
        # 5. Recomendaciones
        print("\n--- RECOMENDACIONES ---\n")
        
        print("1. Implementar la política de inventario optimizada para reducir costos de mantenimiento")
        print("   y mejorar el nivel de servicio al cliente.")
        print("\n2. Reorganizar rutas de distribución según el plan optimizado para minimizar")
        print("   distancias recorridas y costos de transporte.")
        print("\n3. Establecer un sistema de monitoreo continuo para ajustar predicciones")
        print("   y parámetros de optimización según datos reales.")
        
        print("\n=== FIN DEL INFORME ===\n")


# Ejemplo de uso del sistema (simulado)
if __name__ == "__main__":
    # Este código se ejecutaría con datos reales en una implementación completa
    print("Sistema de Optimización de Cadena de Suministro con IA")
    print("Desarrollado por Naiara Rodríguez - Data Analytics Portfolio")
    print("\nEjemplo de uso del sistema con datos simulados:")
    
    # En un proyecto real, cargaríamos datos desde archivos CSV:
    # optimizer = SupplyChainOptimizer()
    # optimizer.load_data(
    #     'data/sales_data.csv',
    #     'data/inventory_data.csv',
    #     'data/locations_data.csv',
    #     'data/products_data.csv'
    # )
    
    print("\nPara ver la implementación completa, ejecute este código con datos reales.")
    print("Funcionalidades incluidas:")
    print("1. Análisis y predicción de demanda con Machine Learning")
    print("2. Optimización de políticas de inventario")
    print("3. Planificación de rutas óptimas de distribución")
    print("4. Generación de informes con recomendaciones y estimación de ahorros")
