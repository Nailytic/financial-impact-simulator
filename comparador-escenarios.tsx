import React, { useState } from 'react';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ComposedChart, Area } from 'recharts';
import { BarChart2, Sliders, Save, FileText, Download } from 'lucide-react';

const ComparadorEscenarios = () => {
  // Estados para manejar los escenarios y configuraciones
  const [tipoGrafica, setTipoGrafica] = useState('linea');
  const [metricas, setMetricas] = useState(['ingresos', 'costos', 'beneficio']);
  const [periodoProyeccion, setPeriodoProyeccion] = useState(24);
  const [escenarioActivo, setEscenarioActivo] = useState('todos');
  
  // Escenarios predefinidos
  const escenarios = [
    {
      nombre: 'Escenario Base',
      descripcion: 'Proyección con las condiciones actuales del mercado',
      color: '#8884d8',
      visible: true
    },
    {
      nombre: 'Expansión de Mercado',
      descripcion: 'Inversión en nuevos mercados geográficos',
      color: '#82ca9d',
      visible: true
    },
    {
      nombre: 'Reducción de Costos',
      descripcion: 'Optimización de procesos y reducción de gastos operativos',
      color: '#ffc658',
      visible: true
    },
    {
      nombre: 'Nuevo Producto',
      descripcion: 'Lanzamiento de nueva línea de productos',
      color: '#ff7300',
      visible: false
    }
  ];

  // Datos simulados para los escenarios (24 meses de proyección)
  const generarDatosProyeccion = () => {
    const datos = [];
    for (let i = 1; i <= periodoProyeccion; i++) {
      const mes = i <= 12 ? `Mes ${i}` : `Mes ${i}`;
      
      const factorBase = 1 + (i * 0.01);
      const factorExpansion = 1 + (i * 0.025);
      const factorReduccion = 1 + (i * 0.008);
      const factorNuevo = i > 6 ? 1 + ((i-6) * 0.04) : 1;
      
      datos.push({
        mes: mes,
        // Escenario Base
        ingresos_base: Math.round(10000 * factorBase),
        costos_base: Math.round(7000 * factorBase),
        beneficio_base: Math.round(3000 * factorBase),
        roi_base: Math.round(30 + (i * 0.2)),
        
        // Escenario Expansión
        ingresos_expansion: Math.round(10000 * factorExpansion),
        costos_expansion: Math.round(8500 * factorExpansion),
        beneficio_expansion: Math.round(10000 * factorExpansion - 8500 * factorExpansion),
        roi_expansion: Math.round(25 + (i * 0.4)),
        
        // Escenario Reducción
        ingresos_reduccion: Math.round(10000 * factorReduccion),
        costos_reduccion: Math.round(6000 * factorReduccion),
        beneficio_reduccion: Math.round(10000 * factorReduccion - 6000 * factorReduccion),
        roi_reduccion: Math.round(35 + (i * 0.3)),
        
        // Escenario Nuevo Producto
        ingresos_nuevo: Math.round(10000 * factorNuevo),
        costos_nuevo: Math.round(7500 * factorNuevo),
        beneficio_nuevo: Math.round(10000 * factorNuevo - 7500 * factorNuevo),
        roi_nuevo: i > 6 ? Math.round(20 + ((i-6) * 0.8)) : 20
      });
    }
    return datos;
  };

  const datos = generarDatosProyeccion();
  
  // Función para togglear la visibilidad de un escenario
  const toggleEscenario = (nombre) => {
    const nuevosEscenarios = escenarios.map(esc => 
      esc.nombre === nombre ? {...esc, visible: !esc.visible} : esc
    );
    // Actualizar estado de escenarios aquí en una implementación real
  };
  
  // Función para filtrar datos según los escenarios activos
  const filtrarDatos = (datos) => {
    // En un caso real, filtraríamos basados en los escenarios visibles
    return datos;
  };
  
  const datosFiltrados = filtrarDatos(datos);
  
  return (
    <div className="bg-gray-50 min-h-screen p-4">
      <div className="bg-white shadow rounded-lg p-4">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-2xl font-semibold text-gray-800">Comparador de Escenarios</h1>
          <div className="flex space-x-2">
            <button className="flex items-center px-3 py-1 bg-blue-600 text-white rounded">
              <Save className="h-4 w-4 mr-1" />
              Guardar
            </button>
            <button className="flex items-center px-3 py-1 bg-gray-200 text-gray-700 rounded">
              <Download className="h-4 w-4 mr-1" />
              Exportar
            </button>
          </div>
        </div>
        
        {/* Panel de configuración */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 mb-6">
          <div className="col-span-1 lg:col-span-4 bg-gray-100 p-4 rounded-lg">
            <div className="flex items-center mb-3">
              <Sliders className="h-5 w-5 text-blue-600 mr-2" />
              <h2 className="text-lg font-medium">Configuración</h2>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Tipo de gráfica */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Tipo de visualización</label>
                <select 
                  className="w-full border border-gray-300 rounded-md px-3 py-2"
                  value={tipoGrafica}
                  onChange={(e) => setTipoGrafica(e.target.value)}
                >
                  <option value="linea">Líneas</option>
                  <option value="barra">Barras</option>
                  <option value="area">Áreas</option>
                  <option value="compuesta">Compuesta</option>
                </select>
              </div>
              
              {/* Métricas */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Métrica principal</label>
                <select 
                  className="w-full border border-gray-300 rounded-md px-3 py-2"
                  value={metricas[0]}
                  onChange={(e) => setMetricas([e.target.value, ...metricas.slice(1)])}
                >
                  <option value="ingresos">Ingresos</option>
                  <option value="costos">Costos</option>
                  <option value="beneficio">Beneficio</option>
                  <option value="roi">ROI (%)</option>
                </select>
              </div>
              
              {/* Período de proyección */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Período de proyección</label>
                <select 
                  className="w-full border border-gray-300 rounded-md px-3 py-2"
                  value={periodoProyeccion}
                  onChange={(e) => setPeriodoProyeccion(parseInt(e.target.value))}
                >
                  <option value="12">12 meses</option>
                  <option value="24">24 meses</option>
                  <option value="36">36 meses</option>
                  <option value="60">5 años</option>
                </select>
              </div>
            </div>
          </div>
        </div>
        
        {/* Selector de escenarios */}
        <div className="mb-6">
          <h2 className="text-lg font-medium mb-3">Escenarios</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
            {escenarios.map((escenario, index) => (
              <div 
                key={index}
                className={`border rounded-lg p-3 cursor-pointer ${escenario.visible ? 'border-blue-500 bg-blue-50' : 'border-gray-300'}`}
                onClick={() => toggleEscenario(escenario.nombre)}
              >
                <div className="flex items-center">
                  <div className="h-4 w-4 rounded-full mr-2" style={{ backgroundColor: escenario.color }}></div>
                  <h3 className="font-medium text-gray-800">{escenario.nombre}</h3>
                </div>
                <p className="text-sm text-gray-600 mt-1">{escenario.descripcion}</p>
              </div>
            ))}
          </div>
        </div>
        
        {/* Visualización principal */}
        <div className="bg-white rounded-lg border shadow p-4">
          <div className="flex items-center mb-4">
            <BarChart2 className="h-5 w-5 text-blue-600 mr-2" />
            <h2 className="text-lg font-medium">Comparativa de escenarios: {metricas[0].charAt(0).toUpperCase() + metricas[0].slice(1)}</h2>
          </div>
          
          <ResponsiveContainer width="100%" height={400}>
            {tipoGrafica === 'linea' && (
              <LineChart data={datosFiltrados} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="mes" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey={`${metricas[0]}_base`} stroke="#8884d8" name="Escenario Base" strokeWidth={2} activeDot={{ r: 8 }} />
                <Line type="monotone" dataKey={`${metricas[0]}_expansion`} stroke="#82ca9d" name="Expansión Mercado" strokeWidth={2} />
                <Line type="monotone" dataKey={`${metricas[0]}_reduccion`} stroke="#ffc658" name="Reducción Costos" strokeWidth={2} />
                <Line type="monotone" dataKey={`${metricas[0]}_nuevo`} stroke="#ff7300" name="Nuevo Producto" strokeWidth={2} />
              </LineChart>
            )}
            
            {tipoGrafica === 'barra' && (
              <BarChart data={datosFiltrados} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="mes" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey={`${metricas[0]}_base`} fill="#8884d8" name="Escenario Base" />
                <Bar dataKey={`${metricas[0]}_expansion`} fill="#82ca9d" name="Expansión Mercado" />
                <Bar dataKey={`${metricas[0]}_reduccion`} fill="#ffc658" name="Reducción Costos" />
                <Bar dataKey={`${metricas[0]}_nuevo`} fill="#ff7300" name="Nuevo Producto" />
              </BarChart>
            )}
            
            {tipoGrafica === 'area' && (
              <BarChart data={datosFiltrados} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="mes" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey={`${metricas[0]}_base`} stackId="a" fill="#8884d8" name="Escenario Base" />
                <Bar dataKey={`${metricas[0]}_expansion`} stackId="a" fill="#82ca9d" name="Expansión Mercado" />
                <Bar dataKey={`${metricas[0]}_reduccion`} stackId="a" fill="#ffc658" name="Reducción Costos" />
                <Bar dataKey={`${metricas[0]}_nuevo`} stackId="a" fill="#ff7300" name="Nuevo Producto" />
              </BarChart>
            )}
            
            {tipoGrafica === 'compuesta' && (
              <ComposedChart data={datosFiltrados} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="mes" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey={`${metricas[0]}_base`} fill="#8884d8" name="Escenario Base" />
                <Bar dataKey={`${metricas[0]}_expansion`} fill="#82ca9d" name="Expansión Mercado" />
                <Bar dataKey={`${metricas[0]}_reduccion`} fill="#ffc658" name="Reducción Costos" />
                <Line type="monotone" dataKey={`${metricas[0]}_nuevo`} stroke="#ff7300" name="Nuevo Producto" strokeWidth={2} />
              </ComposedChart>
            )}
          </ResponsiveContainer>
        </div>
        
        {/* Tabla comparativa de resultados */}
        <div className="mt-6">
          <h2 className="text-lg font-medium mb-3">Resultados clave por escenario</h2>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Escenario</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">ROI Promedio</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Beneficio Total</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Ingresos Acum.</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Costos Acum.</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <div className="h-3 w-3 rounded-full bg-blue-600 mr-2"></div>
                      <div className="text-sm font-medium text-gray-900">Escenario Base</div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-900">31.4%</div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-900">€78,450</div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-900">€261,500</div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-900">€183,050</div>
                  </td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <div className="h-3 w-3 rounded-full bg-green-500 mr-2"></div>
                      <div className="text-sm font-medium text-gray-900">Expansión de Mercado</div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-900">28.7%</div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-900">€85,780</div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-900">€298,400</div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-900">€212,620</div>
                  </td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <div className="h-3 w-3 rounded-full bg-yellow-500 mr-2"></div>
                      <div className="text-sm font-medium text-gray-900">Reducción de Costos</div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-900">37.8%</div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-900">€104,350</div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-900">€260,880</div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-900">€156,530</div>
                  </td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <div className="h-3 w-3 rounded-full bg-orange-500 mr-2"></div>
                      <div className="text-sm font-medium text-gray-900">Nuevo Producto</div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-900">33.5%</div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-900">€96,270</div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-900">€289,430</div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-900">€193,160</div>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ComparadorEscenarios;