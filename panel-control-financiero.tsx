import React, { useState } from 'react';
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { TrendingUp, TrendingDown, AlertCircle, DollarSign, BarChart2, PieChart, Calendar } from 'lucide-react';

const DashboardFinanciero = () => {
  // Datos simulados para las visualizaciones
  const [periodoSeleccionado, setPeriodoSeleccionado] = useState('Anual');
  
  // Datos KPIs
  const kpiData = [
    { nombre: 'Ingresos Totales', valor: '€3.45M', cambio: '+12.3%', tendencia: 'up', color: 'text-green-600' },
    { nombre: 'Margen Operativo', valor: '28.5%', cambio: '+3.5%', tendencia: 'up', color: 'text-green-600' },
    { nombre: 'ROI', valor: '18.2%', cambio: '-1.2%', tendencia: 'down', color: 'text-red-600' },
    { nombre: 'Liquidez', valor: '1.8', cambio: '+0.2', tendencia: 'up', color: 'text-green-600' }
  ];

  // Datos para la gráfica de tendencias
  const dataTendencias = [
    { mes: 'Ene', ingresos: 2400, gastos: 1800, beneficio: 600 },
    { mes: 'Feb', ingresos: 2100, gastos: 1700, beneficio: 400 },
    { mes: 'Mar', ingresos: 2800, gastos: 2000, beneficio: 800 },
    { mes: 'Abr', ingresos: 2700, gastos: 1900, beneficio: 800 },
    { mes: 'May', ingresos: 3100, gastos: 2100, beneficio: 1000 },
    { mes: 'Jun', ingresos: 3300, gastos: 2300, beneficio: 1000 },
    { mes: 'Jul', ingresos: 3500, gastos: 2400, beneficio: 1100 },
    { mes: 'Ago', ingresos: 3200, gastos: 2200, beneficio: 1000 },
    { mes: 'Sep', ingresos: 3400, gastos: 2300, beneficio: 1100 },
    { mes: 'Oct', ingresos: 3800, gastos: 2500, beneficio: 1300 },
    { mes: 'Nov', ingresos: 4000, gastos: 2600, beneficio: 1400 },
    { mes: 'Dic', ingresos: 4500, gastos: 2800, beneficio: 1700 }
  ];

  // Datos para el desglose de gastos
  const dataGastos = [
    { name: 'Personal', value: 45 },
    { name: 'Marketing', value: 20 },
    { name: 'Operaciones', value: 15 },
    { name: 'I+D', value: 12 },
    { name: 'Administración', value: 8 }
  ];

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

  // Datos para proyecciones
  const dataProyecciones = [
    { mes: 'Ene', actual: 2400, proyectado: 2500 },
    { mes: 'Feb', actual: 2100, proyectado: 2200 },
    { mes: 'Mar', actual: 2800, proyectado: 2700 },
    { mes: 'Abr', actual: 2700, proyectado: 2800 },
    { mes: 'May', actual: 3100, proyectado: 3000 },
    { mes: 'Jun', actual: 3300, proyectado: 3200 },
    { mes: 'Jul', actual: 3500, proyectado: 3600 },
    { mes: 'Ago', actual: 3200, proyectado: 3400 },
    { mes: 'Sep', actual: 3400, proyectado: 3600 },
    { mes: 'Oct', actual: 3800, proyectado: 3900 },
    { mes: 'Nov', actual: 4000, proyectado: 4200 },
    { mes: 'Dic', actual: null, proyectado: 4500 }
  ];

  return (
    <div className="bg-gray-50 min-h-screen p-4">
      <div className="bg-white shadow rounded-lg p-4 mb-6">
        <div className="flex justify-between items-center mb-4">
          <h1 className="text-2xl font-semibold text-gray-800">Panel de Control Financiero</h1>
          <div className="flex space-x-2">
            <button 
              className={`px-3 py-1 rounded ${periodoSeleccionado === 'Mensual' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-700'}`}
              onClick={() => setPeriodoSeleccionado('Mensual')}
            >
              Mensual
            </button>
            <button 
              className={`px-3 py-1 rounded ${periodoSeleccionado === 'Trimestral' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-700'}`}
              onClick={() => setPeriodoSeleccionado('Trimestral')}
            >
              Trimestral
            </button>
            <button 
              className={`px-3 py-1 rounded ${periodoSeleccionado === 'Anual' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-700'}`}
              onClick={() => setPeriodoSeleccionado('Anual')}
            >
              Anual
            </button>
          </div>
        </div>
        
        {/* Tarjetas de KPIs */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          {kpiData.map((kpi, index) => (
            <div key={index} className="bg-white rounded-lg border shadow p-4">
              <div className="flex justify-between items-center">
                <h3 className="text-sm font-medium text-gray-500">{kpi.nombre}</h3>
                {kpi.tendencia === 'up' ? (
                  <TrendingUp className="h-5 w-5 text-green-500" />
                ) : (
                  <TrendingDown className="h-5 w-5 text-red-500" />
                )}
              </div>
              <div className="mt-2">
                <p className="text-2xl font-semibold">{kpi.valor}</p>
                <p className={`text-sm ${kpi.color}`}>{kpi.cambio} vs. período anterior</p>
              </div>
            </div>
          ))}
        </div>
        
        {/* Gráfica de Tendencias */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-lg border shadow p-4">
            <div className="flex items-center mb-4">
              <BarChart2 className="h-5 w-5 text-blue-600 mr-2" />
              <h2 className="text-lg font-medium">Evolución de Ingresos y Gastos</h2>
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={dataTendencias} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="mes" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Area type="monotone" dataKey="ingresos" stackId="1" stroke="#8884d8" fill="#8884d8" />
                <Area type="monotone" dataKey="gastos" stackId="1" stroke="#82ca9d" fill="#82ca9d" />
                <Area type="monotone" dataKey="beneficio" stackId="2" stroke="#ffc658" fill="#ffc658" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          
          {/* Desglose de Gastos */}
          <div className="bg-white rounded-lg border shadow p-4">
            <div className="flex items-center mb-4">
              <PieChart className="h-5 w-5 text-blue-600 mr-2" />
              <h2 className="text-lg font-medium">Desglose de Gastos</h2>
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={dataGastos}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                >
                  {dataGastos.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
        
        {/* Proyecciones Financieras */}
        <div className="bg-white rounded-lg border shadow p-4 mt-6">
          <div className="flex items-center mb-4">
            <Calendar className="h-5 w-5 text-blue-600 mr-2" />
            <h2 className="text-lg font-medium">Proyecciones vs. Resultados Actuales</h2>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={dataProyecciones} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="mes" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="actual" stroke="#8884d8" strokeWidth={2} dot={{ r: 4 }} name="Resultados Actuales" />
              <Line type="monotone" dataKey="proyectado" stroke="#82ca9d" strokeWidth={2} strokeDasharray="5 5" dot={{ r: 4 }} name="Proyección" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

export default DashboardFinanciero;