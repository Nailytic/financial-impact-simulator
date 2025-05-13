import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import calendar
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(page_title="Dashboard de Productividad y Bienestar Laboral", 
                   layout="wide", 
                   initial_sidebar_state="expanded")

# Funciones para generar y procesar datos de ejemplo
def generate_sample_data(n_employees=100, days=90):
    """
    Genera datos sintéticos de productividad y bienestar laboral para demostración
    """
    np.random.seed(42)
    
    # Fechas de los últimos días
    end_date = datetime.datetime.now().date()
    start_date = end_date - datetime.timedelta(days=days-1)
    dates = [start_date + datetime.timedelta(days=i) for i in range(days)]
    
    # IDs de empleados
    employee_ids = [f"EMP{i:03d}" for i in range(1, n_employees+1)]
    
    # Departamentos
    departments = ['Desarrollo', 'Marketing', 'Ventas', 'Soporte', 'Recursos Humanos', 'Finanzas']
    employee_departments = np.random.choice(departments, size=n_employees)
    
    # Roles
    roles = {
        'Desarrollo': ['Desarrollador Senior', 'Desarrollador Junior', 'Arquitecto', 'QA', 'DevOps'],
        'Marketing': ['Especialista SEO', 'Diseñador', 'Especialista en Redes Sociales', 'Content Manager'],
        'Ventas': ['Account Manager', 'Sales Representative', 'Sales Manager'],
        'Soporte': ['Soporte Técnico', 'Customer Success', 'Technical Writer'],
        'Recursos Humanos': ['HR Specialist', 'Recruiter', 'HR Manager'],
        'Finanzas': ['Accountant', 'Financial Analyst', 'Controller']
    }
    
    employee_roles = [np.random.choice(roles[dept]) for dept in employee_departments]
    
    # Años de experiencia
    experience_years = np.random.randint(1, 15, size=n_employees)
    
    # Crear dataframe base de empleados
    employees_df = pd.DataFrame({
        'employee_id': employee_ids,
        'department': employee_departments,
        'role': employee_roles,
        'experience_years': experience_years
    })
    
    # Crear patrones base para cada empleado
    employee_patterns = {}
    for emp_id in employee_ids:
        # Productividad base (varía por empleado)
        base_productivity = np.random.normal(70, 15)
        # Nivel de estrés base
        base_stress = np.random.normal(40, 15)
        # Satisfacción base
        base_satisfaction = np.random.normal(75, 15)
        # Horas de trabajo base
        base_work_hours = np.random.normal(8, 1)
        
        employee_patterns[emp_id] = {
            'base_productivity': base_productivity,
            'base_stress': base_stress,
            'base_satisfaction': base_satisfaction,
            'base_work_hours': base_work_hours
        }
    
    # Crear dataframe diario
    data = []
    
    for date in dates:
        # Factores del día (afecta a todos los empleados)
        day_factor = np.random.normal(1, 0.05)  # Factor aleatorio diario
        day_of_week = date.weekday()  # 0=Lunes, 6=Domingo
        
        # Patrón semanal (menor productividad en lunes, mayor el miércoles-jueves, baja el viernes)
        weekday_productivity_factor = {
            0: 0.95,  # Lunes
            1: 1.0,   # Martes
            2: 1.05,  # Miércoles
            3: 1.05,  # Jueves
            4: 0.97,  # Viernes
            5: 0.85,  # Sábado (si trabajan)
            6: 0.8,   # Domingo (si trabajan)
        }
        
        # Patrón estacional (mes)
        month = date.month
        # Menor productividad en verano y cerca de navidad
        if month in [7, 8, 12]:
            seasonal_factor = 0.95
        elif month in [1, 9]:  # Mayor después de vacaciones
            seasonal_factor = 1.05
        else:
            seasonal_factor = 1.0
            
        # Verificar si es fin de semana
        is_weekend = day_of_week >= 5
        
        for idx, employee in employees_df.iterrows():
            emp_id = employee['employee_id']
            
            # Si es fin de semana, la mayoría no trabaja
            if is_weekend and np.random.random() > 0.2:  # Solo 20% trabaja en fin de semana
                continue
                
            patterns = employee_patterns[emp_id]
            
            # Calcular métricas diarias
            
            # Productividad (tareas completadas/calidad)
            productivity = patterns['base_productivity'] * day_factor * weekday_productivity_factor[day_of_week] * seasonal_factor
            productivity = np.clip(productivity + np.random.normal(0, 5), 0, 100)
            
            # Horas trabajadas
            work_hours = patterns['base_work_hours']
            # Añadir variación
            if is_weekend:
                work_hours *= 0.8  # Menos horas en fin de semana
            work_hours = np.clip(work_hours + np.random.normal(0, 0.5), 4, 12)
            
            # Tiempo en reuniones (horas)
            meeting_hours = work_hours * np.random.uniform(0.1, 0.3)
            if day_of_week == 0:  # Más reuniones los lunes
                meeting_hours *= 1.5
            meeting_hours = np.clip(meeting_hours, 0, work_hours * 0.7)
            
            # Concentración (tiempo productivo)
            focus_hours = work_hours - meeting_hours - np.random.uniform(0.5, 1.5)  # Tiempo perdido
            focus_hours = np.clip(focus_hours, 0, work_hours)
            
            # Nivel de estrés
            stress_level = patterns['base_stress']
            # Factores que aumentan el estrés
            stress_factors = 1.0
            if work_hours > 9:  # Sobrecarga
                stress_factors += (work_hours - 9) * 0.15
            if meeting_hours / work_hours > 0.4:  # Muchas reuniones
                stress_factors += 0.2
            if day_of_week == 4:  # Viernes, entregas
                stress_factors += 0.1
                
            stress_level *= stress_factors
            stress_level = np.clip(stress_level + np.random.normal(0, 5), 0, 100)
            
            # Satisfacción
            satisfaction = patterns['base_satisfaction']
            # Factores que afectan la satisfacción
            if stress_level > 70:  # Alto estrés reduce satisfacción
                satisfaction *= 0.85
            if productivity > 80:  # Alta productividad aumenta satisfacción
                satisfaction *= 1.1
            if focus_hours / work_hours < 0.4:  # Poco tiempo productivo
                satisfaction *= 0.9
                
            satisfaction = np.clip(satisfaction + np.random.normal(0, 3), 0, 100)
            
            # Equilibrio trabajo-vida (Work-Life Balance)
            if work_hours <= 8:
                work_life_balance = np.random.uniform(70, 90)
            else:
                # Disminuye con horas extras
                work_life_balance = np.random.uniform(50, 80) - (work_hours - 8) * 7
                
            work_life_balance = np.clip(work_life_balance + np.random.normal(0, 5), 0, 100)
            
            # Compromiso (Engagement)
            engagement = (satisfaction * 0.5 + productivity * 0.3 + (100 - stress_level) * 0.2)
            engagement = np.clip(engagement + np.random.normal(0, 5), 0, 100)
            
            # Colaboración (interacciones con colegas)
            collaboration = meeting_hours * np.random.uniform(10, 15)
            collaboration = np.clip(collaboration + np.random.normal(0, 10), 0, 100)
            
            # Agregar registro
            data.append({
                'date': date,
                'employee_id': emp_id,
                'department': employee['department'],
                'role': employee['role'],
                'experience_years': employee['experience_years'],
                'productivity': round(productivity, 1),
                'work_hours': round(work_hours, 1),
                'meeting_hours': round(meeting_hours, 1),
                'focus_hours': round(focus_hours, 1),
                'stress_level': round(stress_level, 1),
                'satisfaction': round(satisfaction, 1),
                'work_life_balance': round(work_life_balance, 1),
                'engagement': round(engagement, 1),
                'collaboration': round(collaboration, 1)
            })
    
    # Crear DataFrame
    df = pd.DataFrame(data)
    
    # Añadir eventos aleatorios (simulando eventos que afectan a la productividad)
    # 1. Sprint deadlines (aumenta estrés y productividad)
    sprint_days = [start_date + datetime.timedelta(days=i) for i in range(days) if (i+1) % 14 == 0]
    for sprint_day in sprint_days:
        # Afecta principalmente a desarrolladores
        dev_df = df[(df['date'] == sprint_day) & (df['department'] == 'Desarrollo')]
        for idx in dev_df.index:
            df.loc[idx, 'stress_level'] = min(100, df.loc[idx, 'stress_level'] * 1.3)
            df.loc[idx, 'productivity'] = min(100, df.loc[idx, 'productivity'] * 1.15)
            df.loc[idx, 'work_hours'] = min(12, df.loc[idx, 'work_hours'] + 1.5)
            df.loc[idx, 'focus_hours'] = min(df.loc[idx, 'work_hours'], df.loc[idx, 'focus_hours'] * 1.2)
    
    # 2. Lanzamiento de producto (afecta a varios departamentos)
    launch_day = start_date + datetime.timedelta(days=days//2)
    launch_period = [launch_day - datetime.timedelta(days=i) for i in range(5)]
    affected_depts = ['Desarrollo', 'Marketing', 'Ventas']
    for day in launch_period:
        affected_df = df[(df['date'] == day) & (df['department'].isin(affected_depts))]
        for idx in affected_df.index:
            # Aumenta estrés, horas y productividad
            df.loc[idx, 'stress_level'] = min(100, df.loc[idx, 'stress_level'] * 1.25)
            df.loc[idx, 'work_hours'] = min(12, df.loc[idx, 'work_hours'] + 2)
            df.loc[idx, 'meeting_hours'] = min(df.loc[idx, 'work_hours'] * 0.6, df.loc[idx, 'meeting_hours'] * 1.5)
            df.loc[idx, 'focus_hours'] = max(0, df.loc[idx, 'work_hours'] - df.loc[idx, 'meeting_hours'] - 1)
            df.loc[idx, 'productivity'] = min(100, df.loc[idx, 'productivity'] * 1.1)
            df.loc[idx, 'work_life_balance'] = max(0, df.loc[idx, 'work_life_balance'] * 0.8)
    
    # 3. Iniciativa de bienestar (mejora satisfacción y equilibrio)
    wellness_start = start_date + datetime.timedelta(days=days//4)
    wellness_period = [wellness_start + datetime.timedelta(days=i) for i in range(14)]
    for day in wellness_period:
        wellness_df = df[df['date'] == day]
        for idx in wellness_df.index:
            if np.random.random() < 0.7:  # 70% participación
                df.loc[idx, 'stress_level'] = max(0, df.loc[idx, 'stress_level'] * 0.9)
                df.loc[idx, 'satisfaction'] = min(100, df.loc[idx, 'satisfaction'] * 1.1)
                df.loc[idx, 'work_life_balance'] = min(100, df.loc[idx, 'work_life_balance'] * 1.15)
    
    return df

def identify_work_profiles(df):
    """
    Identifica perfiles de trabajo basados en patrones de productividad y bienestar
    """
    # Agregación por empleado
    emp_metrics = df.groupby('employee_id').agg({
        'productivity': 'mean',
        'work_hours': 'mean',
        'meeting_hours': 'mean',
        'focus_hours': 'mean',
        'stress_level': 'mean',
        'satisfaction': 'mean',
        'work_life_balance': 'mean',
        'engagement': 'mean',
        'collaboration': 'mean'
    }).reset_index()
    
    # Añadir métricas adicionales
    emp_metrics['focus_ratio'] = emp_metrics['focus_hours'] / emp_metrics['work_hours']
    emp_metrics['meeting_ratio'] = emp_metrics['meeting_hours'] / emp_metrics['work_hours']
    
    # Estandarizar datos para clustering
    features = ['productivity', 'work_hours', 'focus_ratio', 'meeting_ratio', 
                'stress_level', 'satisfaction', 'work_life_balance', 'engagement']
    
    X = emp_metrics[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determinar número óptimo de clusters
    inertia = []
    for k in range(2, 8):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
    
    # Utilizamos 4 clusters (simplificado para este ejemplo)
    k = 4
    kmeans = KMeans(n_clusters=k, random_state=42)
    emp_metrics['work_profile'] = kmeans.fit_predict(X_scaled)
    
    # Analizar características de cada perfil
    profile_descriptions = {}
    
    for profile in range(k):
        profile_data = emp_metrics[emp_metrics['work_profile'] == profile]
        
        # Calcular características promedio
        profile_descriptions[profile] = {
            'count': len(profile_data),
            'avg_productivity': profile_data['productivity'].mean(),
            'avg_work_hours': profile_data['work_hours'].mean(),
            'avg_focus_ratio': profile_data['focus_ratio'].mean(),
            'avg_meeting_ratio': profile_data['meeting_ratio'].mean(),
            'avg_stress': profile_data['stress_level'].mean(),
            'avg_satisfaction': profile_data['satisfaction'].mean(),
            'avg_work_life_balance': profile_data['work_life_balance'].mean(),
            'avg_engagement': profile_data['engagement'].mean()
        }
    
    # Asignar nombres a los perfiles basados en características
    profile_names = {}
    
    for profile, stats in profile_descriptions.items():
        # Lógica para nombrar perfiles
        if stats['avg_productivity'] > 75 and stats['avg_work_life_balance'] > 70:
            profile_names[profile] = "Equilibrado de Alto Rendimiento"
        elif stats['avg_productivity'] > 75 and stats['avg_stress'] > 60:
            profile_names[profile] = "Alto Rendimiento Estresado"
        elif stats['avg_meeting_ratio'] > 0.25 and stats['avg_collaboration'] > 60:
            profile_names[profile] = "Colaborador Extendido"
        elif stats['avg_focus_ratio'] > 0.7:
            profile_names[profile] = "Trabajador Concentrado"
        else:
            profile_names[profile] = f"Perfil {profile+1}"
    
    # Añadir nombre del perfil al dataframe de empleados
    emp_metrics['profile_name'] = emp_metrics['work_profile'].map(profile_names)
    
    # Añadir información del departamento
    dept_info = df.groupby('employee_id')['department'].first().reset_index()
    emp_metrics = emp_metrics.merge(dept_info, on='employee_id')
    
    # Guardar centroides
    centroids = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_), 
        columns=features
    )
    centroids['profile'] = range(k)
    centroids['profile_name'] = centroids['profile'].map(profile_names)
    
    return emp_metrics, centroids, profile_names

def analyze_burnout_risk(df):
    """
    Analiza riesgo de burnout basado en métricas de estrés, 
    satisfacción y equilibrio trabajo-vida
    """
    # Agregación por empleado
    emp_metrics = df.groupby('employee_id').agg({
        'productivity': 'mean',
        'work_hours': 'mean',
        'stress_level': 'mean',
        'satisfaction': 'mean',
        'work_life_balance': 'mean',
        'engagement': 'mean',
        'department': 'first',
        'role': 'first',
        'experience_years': 'first'
    }).reset_index()
    
    # Analizar tendencia de estrés en las últimas 2 semanas
    last_2w = df[df['date'] >= df['date'].max() - datetime.timedelta(days=14)]
    stress_trend = last_2w.groupby(['employee_id', 'date'])['stress_level'].mean().reset_index()
    
    # Calcular pendiente de estrés
    stress_slopes = {}
    for emp in stress_trend['employee_id'].unique():
        emp_data = stress_trend[stress_trend['employee_id'] == emp].sort_values('date')
        if len(emp_data) >= 7:  # Al menos una semana de datos
            x = (emp_data['date'] - emp_data['date'].min()).dt.days.values
            y = emp_data['stress_level'].values
            slope = np.polyfit(x, y, 1)[0]
            stress_slopes[emp] = slope
    
    # Añadir pendiente al dataframe
    emp_metrics['stress_trend'] = emp_metrics['employee_id'].map(stress_slopes).fillna(0)
    
    # Calcular riesgo de burnout
    emp_metrics['burnout_risk_score'] = (
        emp_metrics['stress_level'] * 0.35 +
        (100 - emp_metrics['satisfaction']) * 0.25 +
        (100 - emp_metrics['work_life_balance']) * 0.25 +
        np.clip(emp_metrics['stress_trend'] * 20, 0, 15) +  # Factor de tendencia
        np.clip((emp_metrics['work_hours'] - 8) * 3, 0, 15)  # Factor de sobrecarga
    )
    
    # Normalizar score a 0-100
    emp_metrics['burnout_risk_score'] = np.clip(emp_metrics['burnout_risk_score'], 0, 100)
    
    # Categorizar riesgo
    conditions = [
        (emp_metrics['burnout_risk_score'] < 30),
        (emp_metrics['burnout_risk_score'] < 50),
        (emp_metrics['burnout_risk_score'] < 70),
        (emp_metrics['burnout_risk_score'] >= 70)
    ]
    risk_categories = ['Bajo', 'Moderado', 'Alto', 'Crítico']
    emp_metrics['burnout_risk'] = np.select(conditions, risk_categories)
    
    # Análisis de factores contribuyentes
    emp_metrics['long_hours_factor'] = np.clip((emp_metrics['work_hours'] - 8.5) * 10, 0, 100)
    emp_metrics['stress_factor'] = emp_metrics['stress_level']
    emp_metrics['satisfaction_factor'] = 100 - emp_metrics['satisfaction']
    emp_metrics['balance_factor'] = 100 - emp_metrics['work_life_balance']
    
    # Identificar factor principal para cada empleado
    factors = ['long_hours_factor', 'stress_factor', 'satisfaction_factor', 'balance_factor']
    emp_metrics['primary_factor'] = emp_metrics[factors].idxmax(axis=1)
    emp_metrics['primary_factor'] = emp_metrics['primary_factor'].map({
        'long_hours_factor': 'Horas Extendidas',
        'stress_factor': 'Estrés Elevado',
        'satisfaction_factor': 'Baja Satisfacción',
        'balance_factor': 'Desequilibrio Trabajo-Vida'
    })
    
    return emp_metrics

def analyze_department_metrics(df):
    """
    Analiza métricas de productividad y bienestar por departamento
    """
    # Métrica agregadas por departamento
    dept_metrics = df.groupby('department').agg({
        'productivity': 'mean',
        'work_hours': 'mean',
        'meeting_hours': 'mean',
        'focus_hours': 'mean',
        'stress_level': 'mean',
        'satisfaction': 'mean',
        'work_life_balance': 'mean',
        'engagement': 'mean',
        'collaboration': 'mean',
        'employee_id': 'nunique'
    }).reset_index()
    
    # Renombrar columna para claridad
    dept_metrics = dept_metrics.rename(columns={'employee_id': 'employee_count'})
    
    # Calcular ratios adicionales
    dept_metrics['focus_ratio'] = dept_metrics['focus_hours'] / dept_metrics['work_hours']
    dept_metrics['meeting_ratio'] = dept_metrics['meeting_hours'] / dept_metrics['work_hours']
    
    # Análisis de tendencias por departamento
    dept_trends = df.groupby(['department', 'date']).agg({
        'productivity': 'mean',
        'stress_level': 'mean',
        'satisfaction': 'mean'
    }).reset_index()
    
    # Calcular pendientes para cada departamento (tendencias)
    dept_slopes = {}
    for dept in dept_trends['department'].unique():
        dept_data = dept_trends[dept_trends['department'] == dept].sort_values('date')
        x = (dept_data['date'] - dept_data['date'].min()).dt.days.values
        
        # Pendientes para cada métrica
        prod_slope = np.polyfit(x, dept_data['productivity'].values, 1)[0]
        stress_slope = np.polyfit(x, dept_data['stress_level'].values, 1)[0]
        sat_slope = np.polyfit(x, dept_data['satisfaction'].values, 1)[0]
        
        dept_slopes[dept] = {
            'productivity_trend': prod_slope * 30,  # Escalado a cambio mensual
            'stress_trend': stress_slope * 30,
            'satisfaction_trend': sat_slope * 30
        }
    
    # Añadir tendencias al dataframe
    for metric in ['productivity_trend', 'stress_trend', 'satisfaction_trend']:
        dept_metrics[metric] = dept_metrics['department'].map(
            {dept: slopes[metric] for dept, slopes in dept_slopes.items()}
        )
    
    return dept_metrics, dept_trends

def analyze_optimal_conditions(df):
    """
    Identifica condiciones óptimas para productividad y bienestar
    """
    # Análisis de correlaciones
    correlation_data = df[['productivity', 'work_hours', 'meeting_hours', 'focus_hours', 
                           'stress_level', 'satisfaction', 'work_life_balance', 
                           'engagement', 'collaboration']].copy()
    
    corr_matrix = correlation_data.corr()
    
    # Análisis de condiciones óptimas para productividad
    productivity_factors = []
    
    # Relación entre horas de concentración y productividad
    focus_data = df.groupby('employee_id').agg({
        'focus_hours': 'mean',
        'productivity': 'mean'
    }).reset_index()
    
    focus_corr, _ = pearsonr(focus_data['focus_hours'], focus_data['productivity'])
    productivity_factors.append({
        'factor': 'Horas de Concentración',
        'correlation': focus_corr,
        'optimal_range': f"{focus_data.loc[focus_data['productivity'] > focus_data['productivity'].quantile(0.75), 'focus_hours'].mean():.1f} - {focus_data.loc[focus_data['productivity'] > focus_data['productivity'].quantile(0.9), 'focus_hours'].mean():.1f} horas"
    })
    
    # Relación entre reuniones y productividad
    meeting_data = df.groupby('employee_id').agg({
        'meeting_ratio': lambda x: (df.loc[x.index, 'meeting_hours'] / df.loc[x.index, 'work_hours']).mean(),
        'productivity': 'mean'
    }).reset_index()
    
    meeting_corr, _ = pearsonr(meeting_data['meeting_ratio'], meeting_data['productivity'])
    productivity_factors.append({
        'factor': 'Ratio de Reuniones',
        'correlation': meeting_corr,
        'optimal_range': f"{meeting_data.loc[meeting_data['productivity'] > meeting_data['productivity'].quantile(0.75), 'meeting_ratio'].mean()*100:.1f}% - {meeting_data.loc[meeting_data['productivity'] > meeting_data['productivity'].quantile(0.9), 'meeting_ratio'].mean()*100:.1f}% del tiempo"
    })
    
    # Relación entre estrés y productividad
    stress_data = df.groupby('employee_id').agg({
        'stress_level': 'mean',
        'productivity': 'mean'
    }).reset_index()
    
    stress_corr, _ = pearsonr(stress_data['stress_level'], stress_data['productivity'])
    productivity_factors.append({
        'factor': 'Nivel de Estrés',
        'correlation': stress_corr,
        'optimal_range': f"{stress_data.loc[stress_data['productivity'] > stress_data['productivity'].quantile(0.75), 'stress_level'].mean():.1f} - {stress_data.loc[stress_data['productivity'] > stress_data['productivity'].quantile(0.9), 'stress_level'].mean():.1f}%"
    })
    
    # Análisis de condiciones óptimas para satisfacción
    satisfaction_factors = []
    
    # Relación entre horas de trabajo y satisfacción
    hours_data = df.groupby('employee_id').agg({
        'work_hours': 'mean',
        'satisfaction': 'mean'
    }).reset_index()
    
    hours_corr, _ = pearsonr(hours_data['work_hours'], hours_data['satisfaction'])
    satisfaction_factors.append({
        'factor': 'Horas de Trabajo',
        'correlation': hours_corr,
        'optimal_range': f"{hours_data.loc[hours_data['satisfaction'] > hours_data['satisfaction'].quantile(0.75), 'work_hours'].mean():.1f} - {hours_data.loc[hours_data['satisfaction'] > hours_data['satisfaction'].quantile(0.9), 'work_hours'].mean():.1f} horas"
    })
    
    # Relación entre equilibrio trabajo-vida y satisfacción
    balance_data = df.groupby('employee_id').agg({
        'work_life_balance': 'mean',
        'satisfaction': 'mean'
    }).reset_index()
    
    balance_corr, _ = pearsonr(balance_data['work_life_balance'], balance_data['satisfaction'])
    satisfaction_factors.append({
        'factor': 'Equilibrio Trabajo-Vida',
        'correlation': balance_corr,
        'optimal_range': f"{balance_data.loc[balance_data['satisfaction'] > balance_data['satisfaction'].quantile(0.75), 'work_life_balance'].mean():.1f}+ puntos"
    })
    
    return corr_matrix, productivity_factors, satisfaction_factors

# Función para generar visualizaciones
def create_dashboard_visualizations(df, work_profiles, burnout_risk, dept_metrics, corr_matrix):
    """
    Crea visualizaciones para el dashboard
    """
    visualizations = {}
    
    # 1. Matriz de correlación
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
                vmin=-1, vmax=1, cbar_kws={"shrink": .8})
    plt.title("Correlaciones entre Métricas de Productividad y Bienestar", fontsize=14)
    plt.tight_layout()
    
    # Guardar figura
    fig_correlation = plt.gcf()
    visualizations['correlation_matrix'] = fig_correlation
    
    # 2. Dispersión de Productividad vs. Bienestar con perfiles
    productivity_wellbeing_scatter = px.scatter(
        work_profiles, 
        x='productivity', 
        y='work_life_balance',
        size='work_hours',
        color='profile_name',
        hover_name='employee_id',
        hover_data=['department', 'satisfaction', 'stress_level'],
        title='Relación entre Productividad y Bienestar por Perfil de Trabajo',
        labels={
            'productivity': 'Productividad',
            'work_life_balance': 'Equilibrio Trabajo-Vida',
            'work_hours': 'Horas de Trabajo',
            'profile_name': 'Perfil de Trabajo'
        }
    )
    
    productivity_wellbeing_scatter.update_layout(
        height=600,
        xaxis=dict(range=[0, 100]),
        yaxis=dict(range=[0, 100]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    visualizations['productivity_wellbeing_scatter'] = productivity_wellbeing_scatter
    
    # 3. Mapa de calor de riesgo de burnout por departamento
    burnout_by_dept = burnout_risk.groupby('department').agg({
        'burnout_risk_score': 'mean',
        'employee_id': 'count'
    }).reset_index()
    
    burnout_by_dept = burnout_by_dept.rename(columns={'employee_id': 'employee_count'})
    burnout_by_dept = burnout_by_dept.sort_values('burnout_risk_score', ascending=False)
    
    burnout_heatmap = px.bar(
        burnout_by_dept,
        x='department',
        y='employee_count',
        color='burnout_risk_score',
        color_continuous_scale=[(0, 'green'), (0.4, 'yellow'), (0.7, 'orange'), (1, 'red')],
        title='Riesgo de Burnout por Departamento',
        labels={
            'department': 'Departamento',
            'employee_count': 'Número de Empleados',
            'burnout_risk_score': 'Riesgo de Burnout'
        }
    )
    
    burnout_heatmap.update_layout(
        height=400,
        coloraxis_colorbar=dict(
            title="Riesgo",
            tickvals=[20, 40, 60, 80],
            ticktext=["Bajo", "Moderado", "Alto", "Crítico"]
        )
    )
    
    visualizations['burnout_heatmap'] = burnout_heatmap
    
    # 4. Radar chart de perfiles de trabajo
    profile_metrics = work_profiles.groupby('profile_name').agg({
        'productivity': 'mean',
        'work_hours': 'mean',
        'focus_ratio': 'mean',
        'stress_level': 'mean',
        'satisfaction': 'mean',
        'work_life_balance': 'mean',
        'engagement': 'mean',
        'collaboration': 'mean'
    }).reset_index()
    
    # Normalizar datos para radar chart
    metrics_to_normalize = ['productivity', 'work_hours', 'focus_ratio', 'stress_level', 
                           'satisfaction', 'work_life_balance', 'engagement', 'collaboration']
    
    for metric in metrics_to_normalize:
        max_val = profile_metrics[metric].max()
        min_val = profile_metrics[metric].min()
        profile_metrics[f'{metric}_norm'] = (profile_metrics[metric] - min_val) / (max_val - min_val)
    
    # Crear radar chart
    fig_radar = go.Figure()
    
    categories = ['Productividad', 'Horas de Trabajo', 'Ratio de Concentración', 'Nivel de Estrés',
                 'Satisfacción', 'Equilibrio Trabajo-Vida', 'Compromiso', 'Colaboración']
    
    for i, profile in enumerate(profile_metrics['profile_name']):
        row = profile_metrics[profile_metrics['profile_name'] == profile]
        values = [
            row['productivity_norm'].values[0],
            row['work_hours_norm'].values[0],
            row['focus_ratio_norm'].values[0],
            row['stress_level_norm'].values[0],
            row['satisfaction_norm'].values[0],
            row['work_life_balance_norm'].values[0],
            row['engagement_norm'].values[0],
            row['collaboration_norm'].values[0]
        ]
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=profile
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title='Comparación de Perfiles de Trabajo',
        height=500
    )
    
    visualizations['profile_radar'] = fig_radar
    
    # 5. Gráfico de tendencias por departamento
    dept_metrics_sorted = dept_metrics.sort_values('productivity', ascending=False)
    
    # Crear figura con subplots
    dept_trends = make_subplots(rows=3, cols=1, 
                               subplot_titles=("Productividad por Departamento", 
                                               "Satisfacción por Departamento", 
                                               "Estrés por Departamento"),
                               vertical_spacing=0.1,
                               shared_xaxes=True)
    
    # Añadir barras para métricas actuales
    dept_trends.add_trace(
        go.Bar(
            x=dept_metrics_sorted['department'],
            y=dept_metrics_sorted['productivity'],
            text=dept_metrics_sorted['productivity'].round(1),
            textposition='auto',
            name='Productividad Actual',
            marker_color='royalblue'
        ),
        row=1, col=1
    )
    
    dept_trends.add_trace(
        go.Bar(
            x=dept_metrics_sorted['department'],
            y=dept_metrics_sorted['satisfaction'],
            text=dept_metrics_sorted['satisfaction'].round(1),
            textposition='auto',
            name='Satisfacción Actual',
            marker_color='mediumseagreen'
        ),
        row=2, col=1
    )
    
    dept_trends.add_trace(
        go.Bar(
            x=dept_metrics_sorted['department'],
            y=dept_metrics_sorted['stress_level'],
            text=dept_metrics_sorted['stress_level'].round(1),
            textposition='auto',
            name='Estrés Actual',
            marker_color='salmon'
        ),
        row=3, col=1
    )
    
    # Añadir indicadores de tendencia
    for i, dept in enumerate(dept_metrics_sorted['department']):
        row_data = dept_metrics_sorted[dept_metrics_sorted['department'] == dept]
        
        # Productividad
        trend = row_data['productivity_trend'].values[0]
        if abs(trend) > 0.5:  # Solo mostrar tendencias significativas
            icon = "△" if trend > 0 else "▽"
            dept_trends.add_annotation(
                x=dept, y=row_data['productivity'].values[0] + 2,
                text=f"{icon} {abs(trend):.1f}",
                showarrow=False,
                font=dict(color="green" if trend > 0 else "red"),
                row=1, col=1
            )
        
        # Satisfacción
        trend = row_data['satisfaction_trend'].values[0]
        if abs(trend) > 0.5:  # Solo mostrar tendencias significativas
            icon = "△" if trend > 0 else "▽"
            dept_trends.add_annotation(
                x=dept, y=row_data['satisfaction'].values[0] + 2,
                text=f"{icon} {abs(trend):.1f}",
                showarrow=False,
                font=dict(color="green" if trend > 0 else "red"),
                row=2, col=1
            )
        
        # Estrés
        trend = row_data['stress_trend'].values[0]
        if abs(trend) > 0.5:  # Solo mostrar tendencias significativas
            icon = "△" if trend > 0 else "▽"
            dept_trends.add_annotation(
                x=dept, y=row_data['stress_level'].values[0] + 2,
                text=f"{icon} {abs(trend):.1f}",
                showarrow=False,
                font=dict(color="red" if trend > 0 else "green"),
                row=3, col=1
            )
    
    dept_trends.update_layout(
        height=700,
        showlegend=False,
        title_text="Métricas por Departamento con Tendencias Mensuales"
    )
    
    visualizations['dept_trends'] = dept_trends
    
    # 6. Diagrama de flujo focus-productivity
    focus_prod_data = df.groupby('employee_id').agg({
        'focus_hours': 'mean',
        'productivity': 'mean',
        'department': 'first'
    }).reset_index()
    
    focus_flow = px.scatter(
        focus_prod_data,
        x='focus_hours',
        y='productivity',
        color='department',
        trendline='ols',
        title='Relación entre Horas de Concentración y Productividad',
        labels={
            'focus_hours': 'Horas de Concentración Diarias',
            'productivity': 'Productividad',
            'department': 'Departamento'
        }
    )
    
    focus_flow.update_layout(height=450)
    
    visualizations['focus_flow'] = focus_flow
    
    # 7. Predictor de burnout
    # Top empleados en riesgo
    high_risk = burnout_risk[burnout_risk['burnout_risk_score'] > 60].sort_values('burnout_risk_score', ascending=False)
    
    if len(high_risk) > 0:
        risk_factors = pd.melt(
            high_risk[['employee_id', 'long_hours_factor', 'stress_factor', 'satisfaction_factor', 'balance_factor']],
            id_vars=['employee_id'],
            var_name='factor',
            value_name='score'
        )
        
        # Mapear nombres de factores
        risk_factors['factor'] = risk_factors['factor'].map({
            'long_hours_factor': 'Horas Extendidas',
            'stress_factor': 'Estrés Elevado',
            'satisfaction_factor': 'Baja Satisfacción',
            'balance_factor': 'Desequilibrio Trabajo-Vida'
        })
        
        burnout_prediction = px.bar(
            risk_factors,
            x='score',
            y='employee_id',
            color='factor',
            orientation='h',
            title='Factores de Riesgo de Burnout para Empleados en Alerta',
            labels={
                'employee_id': 'Empleado',
                'score': 'Puntuación del Factor',
                'factor': 'Factor de Riesgo'
            }
        )
        
        burnout_prediction.update_layout(
            height=min(100 + len(high_risk) * 30, 600),
            yaxis={'categoryorder':'total ascending'}
        )
        
        visualizations['burnout_prediction'] = burnout_prediction
    
    return visualizations

# Aplicación principal
def main():
    # Título y descripción
    st.title("Dashboard de Productividad y Bienestar Laboral")
    st.markdown("""
    Este dashboard analiza la relación entre productividad y bienestar laboral, 
    identificando patrones, perfiles de trabajo y factores de riesgo.
    """)
    
    # Sidebar para controles
    st.sidebar.header("Controles")
    
    # Opción para usar datos generados o cargar propios (simulado)
    data_option = st.sidebar.radio(
        "Fuente de datos:",
        ["Datos simulados", "Cargar mis datos (demo)"]
    )
    
    if data_option == "Datos simulados":
        # Parámetros para datos generados
        n_employees = st.sidebar.slider("Número de empleados", 20, 200, 100)
        days = st.sidebar.slider("Período de análisis (días)", 30, 180, 90)
        
        # Generar datos de ejemplo
        with st.spinner("Generando datos de ejemplo..."):
            df = generate_sample_data(n_employees, days)
            st.sidebar.success(f"Datos generados: {len(df)} registros")
    else:
        # Simulación de carga de datos
        uploaded_file = st.sidebar.file_uploader("Cargar archivo CSV", type=["csv"])
        
        if uploaded_file is not None:
            # En una implementación real, cargaríamos el archivo
            # Para demo, usamos datos generados
            with st.spinner("Cargando datos..."):
                df = generate_sample_data(100, 90)
                st.sidebar.success(f"Datos cargados: {len(df)} registros")
        else:
            st.info("Por favor, carga un archivo CSV o selecciona 'Datos simulados'")
            st.stop()
    
    # Filtros para el dashboard
    st.sidebar.header("Filtros")
    
    # Filtro de departamento
    all_departments = sorted(df['department'].unique())
    selected_departments = st.sidebar.multiselect(
        "Departamentos",
        all_departments,
        default=all_departments
    )
    
    # Filtro de fechas
    min_date = df['date'].min()
    max_date = df['date'].max()
    date_range = st.sidebar.date_input(
        "Rango de fechas",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = df[(df['date'] >= pd.Timestamp(start_date)) & 
                        (df['date'] <= pd.Timestamp(end_date))]
    else:
        filtered_df = df
    
    # Aplicar filtro de departamento
    if selected_departments:
        filtered_df = filtered_df[filtered_df['department'].isin(selected_departments)]
    
    if len(filtered_df) == 0:
        st.warning("No hay datos para los filtros seleccionados. Por favor, ajusta los filtros.")
        st.stop()
    
    # Procesar datos
    with st.spinner("Analizando datos..."):
        # Identificar perfiles de trabajo
        work_profiles, centroids, profile_names = identify_work_profiles(filtered_df)
        
        # Analizar riesgo de burnout
        burnout_risk = analyze_burnout_risk(filtered_df)
        
        # Analizar métricas por departamento
        dept_metrics, dept_trends = analyze_department_metrics(filtered_df)
        
        # Analizar condiciones óptimas
        corr_matrix, productivity_factors, satisfaction_factors = analyze_optimal_conditions(filtered_df)
        
        # Crear visualizaciones
        visualizations = create_dashboard_visualizations(
            filtered_df, work_profiles, burnout_risk, dept_metrics, corr_matrix
        )
    
    # Mostrar KPIs principales
    st.header("Indicadores Clave de Desempeño")
    
    # Crear columnas para KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_productivity = filtered_df['productivity'].mean()
        productivity_trend = np.polyfit(range(len(dept_trends)), dept_trends['productivity'].values, 1)[0] * 30
        
        st.metric(
            "Productividad Promedio",
            f"{avg_productivity:.1f}%",
            f"{productivity_trend:+.1f}% mensual"
        )
    
    with col2:
        avg_satisfaction = filtered_df['satisfaction'].mean()
        satisfaction_trend = np.polyfit(range(len(dept_trends)), dept_trends['satisfaction'].values, 1)[0] * 30
        
        st.metric(
            "Satisfacción",
            f"{avg_satisfaction:.1f}%",
            f"{satisfaction_trend:+.1f}% mensual"
        )
    
    with col3:
        avg_stress = filtered_df['stress_level'].mean()
        stress_trend = np.polyfit(range(len(dept_trends)), dept_trends['stress_level'].values, 1)[0] * 30
        
        st.metric(
            "Nivel de Estrés",
            f"{avg_stress:.1f}%",
            f"{stress_trend:+.1f}% mensual",
            delta_color="inverse"  # Rojo si aumenta
        )
    
    with col4:
        burnout_pct = len(burnout_risk[burnout_risk['burnout_risk_score'] > 60]) / len(burnout_risk) * 100
        
        st.metric(
            "Empleados en Riesgo",
            f"{burnout_pct:.1f}%",
            f"{len(burnout_risk[burnout_risk['burnout_risk_score'] > 60])} empleados"
        )
    
    # Panel de Pestañas
    tab1, tab2, tab3, tab4 = st.tabs([
        "Productividad y Bienestar", 
        "Perfiles de Trabajo",
        "Riesgo de Burnout",
        "Recomendaciones"
    ])
    
    # Pestaña 1: Productividad y Bienestar
    with tab1:
        st.subheader("Relación entre Productividad y Bienestar")
        st.plotly_chart(visualizations['productivity_wellbeing_scatter'], use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Métricas por Departamento")
            st.plotly_chart(visualizations['dept_trends'], use_container_width=True)
        
        with col2:
            st.subheader("Correlación entre Métricas")
            st.pyplot(visualizations['correlation_matrix'])
            
            st.subheader("Impacto de las Horas de Concentración")
            st.plotly_chart(visualizations['focus_flow'], use_container_width=True)
    
    # Pestaña 2: Perfiles de Trabajo
    with tab2:
        st.subheader("Perfiles de Trabajo Identificados")
        
        # Mostrar información de perfiles
        profile_counts = work_profiles['profile_name'].value_counts()
        
        # Columnas para información general
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.plotly_chart(visualizations['profile_radar'], use_container_width=True)
        
        with col2:
            st.subheader("Distribución de Perfiles")
            fig = px.pie(
                values=profile_counts.values,
                names=profile_counts.index,
                title="Distribución de Empleados por Perfil"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Descripción detallada de perfiles
        st.subheader("Características de los Perfiles")
        
        for profile_name in profile_names.values():
            profile_data = work_profiles[work_profiles['profile_name'] == profile_name]
            dept_distribution = profile_data['department'].value_counts().to_dict()
            
            with st.expander(f"Perfil: {profile_name} ({len(profile_data)} empleados)"):
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    metrics = {
                        'Productividad': f"{profile_data['productivity'].mean():.1f}%",
                        'Horas de Trabajo': f"{profile_data['work_hours'].mean():.1f} horas",
                        'Ratio de Concentración': f"{profile_data['focus_ratio'].mean()*100:.1f}%",
                        'Estrés': f"{profile_data['stress_level'].mean():.1f}%",
                        'Satisfacción': f"{profile_data['satisfaction'].mean():.1f}%",
                        'Equilibrio Trabajo-Vida': f"{profile_data['work_life_balance'].mean():.1f}%"
                    }
                    
                    for metric, value in metrics.items():
                        st.text(f"{metric}: {value}")
                
                with col2:
                    st.text("Distribución por Departamento:")
                    for dept, count in dept_distribution.items():
                        st.text(f"- {dept}: {count} empleados ({count/len(profile_data)*100:.1f}%)")
    
    # Pestaña 3: Riesgo de Burnout
    with tab3:
        st.subheader("Análisis de Riesgo de Burnout")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.plotly_chart(visualizations['burnout_heatmap'], use_container_width=True)
        
        with col2:
            # Distribución por nivel de riesgo
            risk_distribution = burnout_risk['burnout_risk'].value_counts().reset_index()
            risk_distribution.columns = ['risk_level', 'count']
            
            # Ordenar por nivel de riesgo
            risk_order = {'Bajo': 0, 'Moderado': 1, 'Alto': 2, 'Crítico': 3}
            risk_distribution['order'] = risk_distribution['risk_level'].map(risk_order)
            risk_distribution = risk_distribution.sort_values('order')
            
            risk_colors = {
                'Bajo': 'green',
                'Moderado': 'yellow',
                'Alto': 'orange',
                'Crítico': 'red'
            }
            
            fig = px.pie(
                risk_distribution,
                values='count',
                names='risk_level',
                title="Distribución de Niveles de Riesgo",
                color='risk_level',
                color_discrete_map=risk_colors
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Factores contribuyentes
        if 'burnout_prediction' in visualizations:
            st.subheader("Factores de Riesgo para Empleados en Alerta")
            st.plotly_chart(visualizations['burnout_prediction'], use_container_width=True)
        
        # Tabla de empleados de alto riesgo
        st.subheader("Empleados con Mayor Riesgo")
        high_risk = burnout_risk[burnout_risk['burnout_risk_score'] > 60].sort_values('burnout_risk_score', ascending=False)
        
        if len(high_risk) > 0:
            high_risk_table = high_risk[['employee_id', 'department', 'role', 'burnout_risk_score', 'burnout_risk', 'primary_factor']].copy()
            high_risk_table.columns = ['ID', 'Departamento', 'Rol', 'Puntuación', 'Nivel de Riesgo', 'Factor Principal']
            
            # Formatear puntuación
            high_risk_table['Puntuación'] = high_risk_table['Puntuación'].round(1)
            
            st.dataframe(high_risk_table, use_container_width=True)
        else:
            st.info("No se encontraron empleados con alto riesgo de burnout.")
    
    # Pestaña 4: Recomendaciones
    with tab4:
        st.subheader("Condiciones Óptimas para Productividad")
        
        prod_recommendations = pd.DataFrame(productivity_factors)
        
        st.write("""
        Según nuestro análisis, estas son las condiciones que muestran la mayor
        correlación con altos niveles de productividad:
        """)
        
        for _, row in prod_recommendations.iterrows():
            factor = row['factor']
            corr = row['correlation']
            opt_range = row['optimal_range']
            
            corr_str = f"{corr:.2f}"
            corr_desc = "positiva fuerte" if corr > 0.7 else "positiva moderada" if corr > 0.3 else \
                       "positiva débil" if corr > 0 else "negativa débil" if corr > -0.3 else \
                       "negativa moderada" if corr > -0.7 else "negativa fuerte"
            
            st.markdown(f"**{factor}** (correlación {corr_str}, {corr_desc})")
            st.markdown(f"Rango óptimo: **{opt_range}**")
            st.markdown("---")
        
        st.subheader("Mejoras Recomendadas por Departamento")
        
        # Generar recomendaciones personalizadas por departamento
        for dept in selected_departments:
            dept_data = dept_metrics[dept_metrics['department'] == dept].iloc[0]
            
            with st.expander(f"Recomendaciones para: {dept}"):
                st.markdown(f"**Situación Actual:**")
                st.markdown(f"- Productividad: {dept_data['productivity']:.1f}%")
                st.markdown(f"- Satisfacción: {dept_data['satisfaction']:.1f}%")
                st.markdown(f"- Nivel de Estrés: {dept_data['stress_level']:.1f}%")
                st.markdown(f"- Horas de Trabajo: {dept_data['work_hours']:.1f} horas")
                st.markdown(f"- Horas de Reuniones: {dept_data['meeting_hours']:.1f} horas ({dept_data['meeting_ratio']*100:.1f}% del tiempo)")
                
                st.markdown("**Recomendaciones Principales:**")
                
                # Recomendaciones basadas en métricas
                if dept_data['meeting_ratio'] > 0.25:
                    st.markdown("🔹 **Reducir tiempo en reuniones:** El departamento pasa un porcentaje elevado del tiempo en reuniones, lo que podría estar afectando la productividad. Considere implementar reuniones más cortas y eficientes.")
                
                if dept_data['focus_ratio'] < 0.6:
                    st.markdown("🔹 **Aumentar tiempo de concentración:** El tiempo dedicado a trabajo concentrado es bajo. Considere implementar bloques de tiempo sin interrupciones o políticas de 'no reuniones' en ciertos días.")
                
                if dept_data['stress_level'] > 60:
                    st.markdown("🔹 **Gestionar niveles de estrés:** Los niveles de estrés son elevados. Considere revisar la carga de trabajo, plazos y expectativas.")
                
                if dept_data['work_hours'] > 8.5:
                    st.markdown("🔹 **Evaluar carga horaria:** Las horas de trabajo están por encima del óptimo. Revisar si hay ineficiencias en procesos o distribución de tareas.")
                
                if dept_data['satisfaction'] < 65:
                    st.markdown("🔹 **Mejorar satisfacción laboral:** Los niveles de satisfacción están por debajo del óptimo. Considere realizar entrevistas individuales para identificar áreas de mejora.")
                
                # Añadir siempre al menos una recomendación
                if not (dept_data['meeting_ratio'] > 0.25 or dept_data['focus_ratio'] < 0.6 or 
                       dept_data['stress_level'] > 60 or dept_data['work_hours'] > 8.5 or 
                       dept_data['satisfaction'] < 65):
                    st.markdown("🔹 **Mantener prácticas actuales:** Las métricas del departamento son generalmente positivas. Considere documentar y compartir buenas prácticas con otros equipos.")
        
        st.subheader("Plan de Acción Recomendado")
        st.markdown("""
        Basado en todos los datos analizados, recomendamos el siguiente plan de acción:
        
        1. **Implementar bloques de concentración:** Designar períodos específicos (mínimo 2 horas diarias) sin reuniones o interrupciones para trabajo concentrado.
        
        2. **Revisar políticas de reuniones:** Limitar reuniones a máximo 25% del tiempo laboral, con agendas claras y duraciones acotadas.
        
        3. **Implementar programa de bienestar:** Sesiones de mindfulness, gestión de estrés y promoción de pausas activas.
        
        4. **Feedback continuo:** Encuestas breves semanales de pulso para monitorear satisfacción y bienestar.
        
        5. **Formación en gestión del tiempo:** Capacitación específica para empleados con patrones de trabajo que muestran riesgos.
        
        6. **Revisión de carga de trabajo:** Especialmente para departamentos con niveles de estrés elevados o tendencias negativas.
        """)
        
        # Métricas sugeridas para seguimiento
        st.subheader("Métricas de Seguimiento Recomendadas")
        
        metrics_col1, metrics_col2 = st.columns(2)
        
        with metrics_col1:
            st.markdown("""
            **Métricas de Productividad:**
            - Ratio de tiempo concentrado vs tiempo total
            - Tiempo efectivo en reuniones
            - Tasa de cumplimiento de plazos
            - Calidad de entregables (revisión por pares)
            - Velocidad de resolución de incidencias
            """)
        
        with metrics_col2:
            st.markdown("""
            **Métricas de Bienestar:**
            - Pulso semanal de satisfacción
            - Autoinforme de nivel de estrés
            - Equilibrio percibido trabajo-vida
            - Tasa de incidentes de salud relacionados con estrés
            - Participación en programas de bienestar
            """)
    
    # Sección final: Insights y conclusiones
    st.header("Insights Principales")
    
    # Obtener algunos insights automáticos basados en los datos
    insights = []
    
    # Insight 1: Relación productividad-concentración
    focus_prod_corr = np.corrcoef(
        df.groupby('employee_id')['focus_hours'].mean(),
        df.groupby('employee_id')['productivity'].mean()
    )[0, 1]
    
    if focus_prod_corr > 0.5:
        insights.append(
            "Existe una fuerte correlación positiva entre el tiempo de concentración y la productividad. "
            f"Por cada hora adicional de trabajo concentrado, la productividad aumenta aproximadamente "
            f"{work_profiles['productivity'].corr(work_profiles['focus_hours']) * 10:.1f} puntos."
        )
    
    # Insight 2: Perfil más productivo
    top_profile = work_profiles.groupby('profile_name')['productivity'].mean().idxmax()
    top_profile_data = work_profiles[work_profiles['profile_name'] == top_profile]
    
    insights.append(
        f"El perfil de trabajo '{top_profile}' muestra la mayor productividad promedio "
        f"({top_profile_data['productivity'].mean():.1f}%), combinando un equilibrio de "
        f"{top_profile_data['work_hours'].mean():.1f} horas de trabajo con un ratio de concentración del "
        f"{top_profile_data['focus_ratio'].mean()*100:.1f}%."
    )
    
    # Insight 3: Riesgo de burnout
    if len(burnout_risk[burnout_risk['burnout_risk_score'] > 60]) > 0:
        high_risk_dept = burnout_risk.groupby('department')['burnout_risk_score'].mean().idxmax()
        insights.append(
            f"El departamento de {high_risk_dept} muestra el mayor riesgo de burnout, con un "
            f"{len(burnout_risk[(burnout_risk['department'] == high_risk_dept) & (burnout_risk['burnout_risk_score'] > 60)])} "
            f"empleados en niveles de riesgo alto o crítico. El factor principal es "
            f"{burnout_risk[burnout_risk['department'] == high_risk_dept]['primary_factor'].mode()[0]}."
        )
    
    # Insight 4: Tendencias
    improving_dept = dept_metrics[dept_metrics['satisfaction_trend'] > 0]['department'].values
    if len(improving_dept) > 0:
        insights.append(
            f"Los departamentos {', '.join(improving_dept)} muestran tendencias positivas "
            f"en satisfacción, lo que sugiere que las iniciativas recientes están teniendo un impacto positivo."
        )
    
    # Mostrar insights
    for i, insight in enumerate(insights, 1):
        st.info(f"**Insight {i}:** {insight}")
    
    # Pie de página
    st.markdown("---")
    st.markdown(
        "Dashboard desarrollado por Naiara Rodríguez - Data Analytics Portfolio"
    )

if __name__ == "__main__":
    main()