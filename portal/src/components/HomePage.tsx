import React from 'react';
import { Brain, Zap, Users, Target, ArrowRight, Play } from 'lucide-react';
import { useIaChat } from '../context/iachatContext';

interface HomePageProps {
  onSectionChange: (section: string) => void;
}

export default function HomePage({ onSectionChange }: HomePageProps) {
  const features = [
    {
      icon: Brain,
      title: 'Aprende IA Generativa',
      description: 'Domina los frameworks más populares como TensorFlow, PyTorch y más'
    },
    {
      icon: Zap,
      title: 'Ejercicios Prácticos',
      description: 'Proyectos reales y casos de uso para aplicar tus conocimientos'
    },
    {
      icon: Users,
      title: 'Asistente IA Personal',
      description: 'Chat integrado que te ayuda 24/7 con dudas y conceptos'
    },
    {
      icon: Target,
      title: 'Progreso Personalizado',
      description: 'Sigue tu avance y recibe recomendaciones adaptadas'
    }
  ];

  const modules = [
    { title: 'Agente simple', duration: '2 horas', level: 'Principiante' },
    { title: 'Agente con Ollama', duration: '3 horas', level: 'Intermedio' },
    { title: 'Agente + MCP', duration: '4 horas', level: 'Avanzado' },
    { title: 'Multiagente (restaurant)', duration: '5 horas', level: 'Experto' }
  ];

  const { changeScope } = useIaChat();

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Hero Section */}
      <div className="text-center mb-16">
        <div className="inline-flex items-center px-4 py-2 bg-blue-50 rounded-full text-blue-700 text-sm font-medium mb-6">
          <Zap className="w-4 h-4 mr-2" />
          Version 0.1
        </div>
        
        <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
          Domina la <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-purple-600">IA Generativa</span>
        </h1>
        
        <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
          Aprende a crear, implementar y optimizar modelos de inteligencia artificial generativa 
          con los frameworks más modernos del mercado.
        </p>
        
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          
          
          <button 
            onClick={() => {
                onSectionChange('chat');
                changeScope('general');
              }} 
            className="inline-flex items-center px-8 py-4 bg-white text-gray-700 font-semibold rounded-xl border-2 border-gray-200 hover:border-gray-300 shadow-sm hover:shadow-md transition-all duration-200"
          >
            Hablar con Asistente
            <ArrowRight className="w-5 h-5 ml-2" />
          </button>
        </div>
      </div>

      

      {/* Learning Path */}
      <div className="bg-white rounded-2xl p-8 shadow-sm border border-gray-100">
        <h2 className="text-3xl font-bold text-gray-900 mb-6 text-center">
          Ruta de Aprendizaje
        </h2>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {modules.map((module, index) => (
            <div key={index} className="relative">
              <div className="bg-gradient-to-br from-gray-50 to-white p-6 rounded-xl border border-gray-200 hover:border-blue-200 transition-all duration-200 hover:shadow-md">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-2xl font-bold text-blue-600">0{index + 1}</span>
                  <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                    module.level === 'Principiante' ? 'bg-green-100 text-green-700' :
                    module.level === 'Intermedio' ? 'bg-yellow-100 text-yellow-700' :
                    module.level === 'Avanzado' ? 'bg-orange-100 text-orange-700' :
                    'bg-red-100 text-red-700'
                  }`}>
                    {module.level}
                  </span>
                </div>
                
                <h3 className="font-semibold text-gray-900 mb-2">{module.title}</h3>
                {/* <p className="text-sm text-gray-500 mb-4">{module.duration}</p> */}
                
                <button 
                  onClick={() => {
                    onSectionChange('chat');
                    changeScope('example'+(index + 1));
                  }}
                  className="w-full py-2 px-4 bg-blue-50 hover:bg-blue-100 text-blue-700 rounded-lg text-sm font-medium transition-colors duration-200">
                  Ver ejemplo asistido 
                </button>
              </div>
              
              {index < modules.length - 1 && (
                <div className="hidden lg:block absolute top-1/2 -right-3 w-6 h-0.5 bg-gray-200"></div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}