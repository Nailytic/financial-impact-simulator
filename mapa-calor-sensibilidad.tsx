import React, { useState } from 'react';
import { Thermometer, ChevronDown, ChevronUp, Download, Save, HelpCircle } from 'lucide-react';
import _ from 'lodash';

const MapaCalorSensibilidad = () => {
  // Estados para manejar las configuraciones
  const [escenarioSeleccionado, setEscenarioSeleccionado] = useState('escenarioBase');
  const [metricaImpacto, setMetricaImpacto] = useState('beneficio');
  const [rangoSensibilidad, setRangoSensibilidad] = useState(20);
  const [ordenarPor, setOrdenarPor] = useState('impacto');
  
  // Datos de variables por categoría
  const categorias = [
    { 
      nombre: 'Variables de Mercado', 
      expandido: true,
      variables: [
        { nombre: 'Tamaño de Mercado', impacto: 83, riesgo: 'alto', categoria: 'mercado' },
        { nombre: 'Participación de Mercado', impacto: 78, riesgo: 'alto', categoria: 'mercado' },
        { nombre: 'Crecimiento del Sector', impacto: 67, riesgo: 'medio', categoria: 'mercado' }
      ]
    },
    { 
      nombre: 'Variables Financieras', 
      expandido: true,
      variables: [
        { nombre: 'Tasa de Interés', impacto: 54, riesgo: 'medio', categoria: 'financiera' },
        { nombre: 'Tipo de Cambio', impacto: 42, riesgo: 'medio', categoria: 'financiera' },
        { nombre: 'Inflación', impacto: 61, riesgo: 'medio', categoria: 'financiera' },
        { nombre: 'Costo de Capital', impacto: 48, riesgo: 'bajo', categoria: 'financiera' }
      ]
    },
    { 
      nombre: 'Variables Operativas', 
      expandido: true,
      variables: [
        { nombre: 'Costo Materia Prima', impacto: 76, riesgo: 'alto', categoria: 'operativa' },
        { nombre: 'Costo Mano de Obra', impacto: 65, riesgo: 'medio', categoria: 'operativa' },
        { nombre: 'Eficiencia Productiva', impacto: 72, riesgo: 'bajo', categoria: 'operativa' },
        { nombre: 'Capacidad Instalada', impacto: 58, riesgo: 'bajo', categoria: 'operativa' }
      ]
    },
    { 
      nombre: 'Marketing y Ventas', 
      expandido: true,
      variables: [
        { nombre: 'Precio de Venta', impacto: 95, riesgo: 'alto', categoria: 'ventas' },
        { nombre: 'Gasto Publicitario', impacto: 51, riesgo: 'medio', categoria: 'ventas' },
        { nombre: 'Fuerza de Ventas', impacto: 63, riesgo: 'medio', categoria: 'ventas' },
        { nombre: 'Promociones', impacto: 46, riesgo: 'bajo', categoria: 'ventas' }
      ]
    },
    { 
      nombre: 'Factores Externos', 
      expandido: false,
      variables: [
        { nombre: 'Regulaciones', impacto: 57, riesgo: 'alto', categoria: 'externo' },
        { nombre: 'Competencia', impacto: 82, riesgo: 'alto', categoria: 'externo' },
        { nombre: 'Factores Climáticos', impacto: 28, riesgo: 'bajo', categoria: 'externo' },
        { nombre: 'Tendencias de Consumo', impacto: 73, riesgo: 'medio', categoria: 'externo' }
      ]
    }
  ];
  
  // Función para obtener el color según el impacto
  const getColorByImpact = (impacto) => {
    if (impacto >= 80) return 'bg-red-600';
    if (impacto >= 70) return 'bg-red-500';
    if (impacto >= 60) return 'bg-orange-500';
    if (impacto >= 50) return 'bg-yellow-500';
    if (impacto >= 40) return 'bg-yellow-400';
    if (impacto >= 30) return 'bg-green-400';
    return 'bg-green-300';
  };
  
  // Función para obtener el texto según el color (para accesibilidad)
  const getTextColorByImpact = (impacto) => {
    if (impacto >= 60) return 'text-white';
    return 'text-gray-800';
  };
  
  // Función para obtener la etiqueta de riesgo
  const getRiskLabel = (riesgo) => {
    switch(riesgo) {
      case 'alto': return <span className="px-2 py-1 bg-red-100 text-red-800 rounded-full text-xs">Alto</span>;
      case 'medio': return <span className="px-2 py-1 bg-yellow-100 text-yellow-800 rounded-full text-xs">Medio</span>;
      case 'bajo': return <span className="px-2 py-1 bg-green-100 text-green-800 rounded-full text-xs">Bajo</span>;
      default: return null;
    }
  };
  
  // Función para ordenar variables
  const ordenarVariables = (variables) => {
    return _.orderBy(variables, [ordenarPor], ['desc']);
  };
  
  // Función para expandir/colapsar categoría
  const toggleCategoria = (index) => {
    const newCategorias = [...categorias];
    newCategorias[index].expandido = !newCategorias[index].expandido;
    // En una implementación real actualizaríamos el estado aquí
  };
  
  // Función para obtener el total de variables
  const getTotalVariables = () => {
    return categorias.reduce((total, cat) => total + cat.variables.length, 0);
  };
  
  // Función para obtener valor del impacto
  const getImpactValue = (impacto) => {
    if (escenarioSeleccionado === 'escenarioOptimista') {
      return impacto * 1.15;
    } else if (escenarioSeleccionado === 'escenarioPesimista') {
      return impacto * 0.85;
    }
    return impacto;
  };
  
  // Variables más impactantes (top 5)
  const getTopVariables = () => {
    const allVariables = categorias.flatMap(cat => cat.variables);
    return _.orderBy(allVariables, ['impacto'], ['desc']).slice(0, 5);
  };
  
  const topVariables = getTopVariables();
  
  return (
    <div className="bg-gray-50 min-h-screen p-4">
      <div className="bg-white shadow rounded-lg p-4">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-2xl font-semibold text-gray-800">Mapa de Calor de Sensibilidad</h1>
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
        <div className="bg-gray-100 p-4 rounded-lg mb-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {/* Escenario */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Escenario</label>
              <select 
                className="w-full border border-gray-300 rounded-md px-3 py-2"
                value={escenarioSeleccionado}
                onChange={(e) => setEscenarioSeleccionado(e.target.value)}
              >
                <option value="escenarioBase">Escenario Base</option>
                <option value="escenarioOptimista">Escenario Optimista</option>
                <option value="escenarioPesimista">Escenario Pesimista</option>
              </select>
            </div>
            
            {/* Métrica de impacto */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Métrica de impacto</label>
              <select 
                className="w-full border border-gray-300 rounded-md px-3 py-2"
                value={metricaImpacto}
                onChange={(e) => setMetricaImpacto(e.target.value)}
              >
                <option value="beneficio">Beneficio Neto</option>
                <option value="ingresos">Ingresos</option>
                <option value="roi">ROI</option>
                <option value="valoracion">Valoración</option>
              </select>
            </div>
            
            {/* Rango de sensibilidad */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Rango de sensibilidad</label>
              <select 
                className="w-full border border-gray-300 rounded-md px-3 py-2"
                value={rangoSensibilidad}
                onChange={(e) => setRangoSensibilidad(parseInt(e.target.value))}
              >
                <option value="10">±10%</option>
                <option value="20">±20%</option>
                <option value="30">±30%</option>
                <option value="50">±50%</option>
              </select>
            </div>
            
            {/* Ordenar por */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Ordenar por</label>
              <select 
                className="w-full border border-gray-300 rounded-md px-3 py-2"
                value={ordenarPor}
                onChange={(e) => setOrdenarPor(e.target.value)}
              >
                <option value="impacto">Impacto</option>
                <option value="nombre">Nombre</option>
                <option value="riesgo">Nivel de Riesgo</option>
              </select>
            </div>
          </div>
        </div>
        
        {/* Resumen y leyenda */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          {/* Resumen */}
          <div className="lg:col-span-2 bg-white rounded-lg border shadow p-4">
            <h2 className="text-lg font-medium mb-3">Variables más impactantes</h2>
            <div className="overflow-x-auto">
              <table className="min-w-full">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Variable</th>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Categoría</th>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Impacto</th>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Riesgo</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {topVariables.map((variable, index) => (
                    <tr key={index}>
                      <td className="px-4 py-2 whitespace-nowrap text-sm font-medium text-gray-900">{variable.nombre}</td>
                      <td className="px-4 py-2 whitespace-nowrap text-sm text-gray-500">
                        {variable.categoria.charAt(0).toUpperCase() + variable.categoria.slice(1)}
                      </td>
                      <td className="px-4 py-2 whitespace-nowrap">
                        <div className="flex items-center">
                          <div className={`h-8 w-8 flex items-center justify-center rounded ${getColorByImpact(variable.impacto)} ${getTextColorByImpact(variable.impacto)}`}>
                            {variable.impacto}
                          </div>
                        </div>
                      </td>
                      <td className="px-4 py-2 whitespace-nowrap">
                        {getRiskLabel(variable.riesgo)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          
          {/* Leyenda */}
          <div className="bg-white rounded-lg border shadow p-4">
            <div className="flex items-center mb-3">
              <Thermometer className="h-5 w-5 text-blue-600 mr-2" />
              <h2 className="text-lg font-medium">Leyenda</h2>
            </div>
            <p className="text-sm text-gray-600 mb-3">
              El mapa de calor muestra la sensibilidad del {metricaImpacto} ante cambios del {rangoSensibilidad}% en cada variable.
            </p>
            <div className="space-y-2">
              <div className="flex items-center">
                <div className="h-6 w-6 bg-red-600 rounded mr-2"></div>
                <span className="text-sm">Impacto muy alto (80-100)</span>
              </div>
              <div className="flex items-center">
                <div className="h-6 w-6 bg-orange-500 rounded mr-2"></div>
                <span className="text-sm">Impacto alto (60-79)</span>
              </div>
              <div className="flex items-center">
                <div className="h-6 w-6 bg-yellow-500 rounded mr-2"></div>
                <span className="text-sm">Impacto medio (40-59)</span>
              </div>
              <div className="flex items-center">
                <div className="h-6 w-6 bg-green-400 rounded mr-2"></div>
                <span className="text-sm">Impacto bajo (20-39)</span>
              </div>
              <div className="flex items-center">
                <div className="h-6 w-6 bg-green-300 rounded mr-2"></div>
                <span className="text-sm">Impacto muy bajo (0-19)</span>
              </div>
            </div>
            <div className="mt-4 pt-4 border-t border-gray-200">
              <p className="text-sm text-gray-700 font-medium">Variables analizadas: {getTotalVariables()}</p>
              <p className="text-sm text-gray-600">Escenario: {escenarioSeleccionado.replace('escenario', '').charAt(0).toUpperCase() + escenarioSeleccionado.replace('escenario', '').slice(1)}</p>
            </div>
          </div>
        </div>
        
        {/* Mapa de calor principal */}
        <div>
          <h2 className="text-lg font-medium mb-3">Mapa de Calor por Categorías</h2>
          <div className="space-y-4">
            {categorias.map((categoria, catIndex) => (
              <div key={catIndex} className="bg-white rounded-lg border shadow overflow-hidden">
                <div 
                  className="flex items-center justify-between bg-gray-50 p-3 cursor-pointer"
                  onClick={() => toggleCategoria(catIndex)}
                >
                  <h3 className="font-medium text-gray-800">{categoria.nombre}</h3>
                  {categoria.expandido ? <ChevronUp className="h-5 w-5 text-gray-500" /> : <ChevronDown className="h-5 w-5 text-gray-500" />}
                </div>
                
                {categoria.expandido && (
                  <div className="p-3">
                    <div className="overflow-x-auto">
                      <table className="min-w-full">
                        <thead className="bg-gray-50">
                          <tr>
                            <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Variable</th>
                            <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Impacto</th>
                            <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Riesgo</th>
                            <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Efecto en {metricaImpacto}</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-200">
                          {ordenarVariables(categoria.variables).map((variable, varIndex) => (
                            <tr key={varIndex}>
                              <td className="px-4 py-2 whitespace-nowrap text-sm font-medium text-gray-900">
                                <div className="flex items-center">
                                  <HelpCircle className="h-4 w-4 text-gray-400 mr-1 cursor-help" title="Ver detalles" />
                                  {variable.nombre}
                                </div>
                              </td>
                              <td className="px-4 py-2 whitespace-nowrap">
                                <div className="flex items-center">
                                  <div 
                                    className={`h-8 w-12 flex items-center justify-center rounded ${getColorByImpact(variable.impacto)} ${getTextColorByImpact(variable.impacto)}`}
                                  >
                                    {variable.impacto}
                                  </div>
                                </div>
                              </td>
                              <td className="px-4 py-2 whitespace-nowrap">
                                {getRiskLabel(variable.riesgo)}
                              </td>
                              <td className="px-4 py-2 whitespace-nowrap">
                                <div className="flex items-center space-x-2">
                                  <div className="h-3 bg-gray-300 rounded-full w-36">
                                    <div 
                                      className={`h-3 rounded-full ${getColorByImpact(variable.impacto)}`}
                                      style={{ width: `${variable.impacto}%` }}
                                    ></div>
                                  </div>
                                  <span className="text-sm text-gray-600">{(variable.impacto/10).toFixed(1)}%</span>
                                </div>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
        
        {/* Análisis y Recomendaciones */}
        <div className="mt-6 bg-white rounded-lg border shadow p-4">
          <h2 className="text-lg font-medium mb-3">Análisis y Recomendaciones</h2>
          <div className="text-sm text-gray-700 space-y-3">
            <p>El análisis de sensibilidad muestra que las variables con mayor impacto en el {metricaImpacto} son <strong>Precio de Venta</strong>, <strong>Tamaño de Mercado</strong> y <strong>Competencia</strong>. Una variación del {rangoSensibilidad}% en estas variables puede cambiar los resultados de manera significativa.</p>
            
            <p>Recomendaciones basadas en este análisis:</p>
            
            <div className="pl-4">
              <p>• <strong>Establecer políticas de precios robustas</strong> con diferentes escenarios y respuestas ágiles a cambios del mercado.</p>
              <p>• <strong>Diversificar mercados</strong> para reducir dependencia de un solo segmento o región geográfica.</p>
              <p>• <strong>Monitorear competencia</strong> de forma continua e implementar estrategias de diferenciación.</p>
              <p>• <strong>Optimizar costos de materias primas</strong> mediante contratos a largo plazo o fuentes alternativas.</p>
            </div>
            
            <p>Para mejorar la resiliencia del modelo financiero, se sugiere realizar análisis de escenarios extremos (stress testing) en las variables de alto impacto y desarrollar planes de contingencia específicos.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MapaCalorSensibilidad;