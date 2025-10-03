import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Lightbulb, BookOpen, Code, Zap } from 'lucide-react';
import { useIaChat } from '../context/iachatContext';
import axios from 'axios';

interface Message {
  id: number;
  content: string;
  isUser: boolean;
  timestamp: Date;
  type?: 'text' | 'suggestion' | 'code';
}

export default function AIChat() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      content: '¡Hola! Soy tu asistente de IA especializado en frameworks de inteligencia artificial generativa. ¿En qué puedo ayudarte hoy?',
      isUser: false,
      timestamp: new Date(),
      type: 'text'
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);


  const quickSuggestions = [
    { icon: BookOpen, text: 'Páginas de la documentación' }
/*     { icon: Code, text: 'Cómo implementar una red neuronal básica' },
    { icon: Zap, text: 'Diferencias entre GPT y BERT' },
    { icon: Lightbulb, text: 'Ideas de proyectos para principiantes' } */
  ];

  const { context, changeScope, addMemoryToScope } = useIaChat();

  // console.log('Contexto actual:', context)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  console.log('Contexto a enviar:' + context.scope)
  console.log(JSON.stringify(JSON.stringify(context.memoryByContext[context.scope])));

  const requestAgentResponse = async (userMessage: string) => {
    // ADD MESSAGE to MEMORY
      addMemoryToScope(userMessage)
    try {
      const response = await axios.post('http://localhost:8081/agent/messages', { message: `${userMessage} **utiliza esta informacion como historico de la conversarion: ${JSON.stringify(context.memoryByContext[context.scope])}**` });
      // console.log('Respuesta del agente:', response.data);
      const ia_response_text = response.data.artifacts[0].parts[0].text;
      const aiResponse: Message = {
        id: Date.now() + 1,
        content: ia_response_text,
        isUser: false,
        timestamp: new Date(),
        type: 'text'
      };
      setMessages(prev => [...prev, aiResponse]);
      setIsTyping(false);

      // ADD RESPONSE to MEMORY
      addMemoryToScope(ia_response_text)
    } catch (error) {
      console.error('Error al obtener respuesta del agente:', error);
      const aiResponse: Message = {
        id: Date.now() + 1,
        content: 'Lo siento, ha ocurrido un error al obtener la respuesta del agente.',
        isUser: false,
        timestamp: new Date(),
        type: 'text'
      };
      setMessages(prev => [...prev, aiResponse]);
      setIsTyping(false);
      addMemoryToScope('Lo siento, ha ocurrido un error al obtener la respuesta del agente.')
    }
  };

  const handleSendMessage = async (message: string = inputValue) => {
    if (!message.trim()) return;

    const userMessage: Message = {
      id: Date.now(),
      content: message,
      isUser: true,
      timestamp: new Date(),
      type: 'text'
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsTyping(true);

    await requestAgentResponse(message);

    // Simular delay de respuesta
    /* setTimeout(() => {
      const aiResponse: Message = {
        id: Date.now() + 1,
        content: generateAIResponse(message),
        isUser: false,
        timestamp: new Date(),
        type: message.toLowerCase().includes('implementar') || message.toLowerCase().includes('código') ? 'code' : 'text'
      };

      setMessages(prev => [...prev, aiResponse]);
      setIsTyping(false);
    }, 1000 + Math.random() * 1000); */

    
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const formatMessageContent = (content: string, type: string = 'text') => {
    if (type === 'code' || content.includes('```')) {
      const parts = content.split(/(```[\s\S]*?```)/g);
      return parts.map((part, index) => {
        if (part.startsWith('```') && part.endsWith('```')) {
          const code = part.slice(3, -3).trim();
          return (
            <pre key={index} className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto my-3">
              <code>{code}</code>
            </pre>
          );
        }
        return <span key={index} className="whitespace-pre-wrap">{part}</span>;
      });
    }

    // Formatear texto con markdown básico
    return content
      .split('\n')
      .map((line, index) => {
        if (line.startsWith('**') && line.endsWith('**')) {
          return <div key={index} className="font-semibold text-gray-900 my-2">{line.slice(2, -2)}</div>;
        }
        if (line.startsWith('• ')) {
          return <div key={index} className="ml-4 my-1">• {line.slice(2)}</div>;
        }
        return <div key={index} className="my-1">{line}</div>;
      });
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 h-[calc(100vh-12rem)] flex flex-col">
        {/* Header */}
        <div className="p-6 border-b border-gray-100">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-blue-600 rounded-full flex items-center justify-center">
              <Bot className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-semibold text-gray-900">Asistente IA</h2>
              <p className="text-sm text-gray-500">Especializado en IA Generativa</p>
            </div>
            <div className="ml-auto">
              <div className="flex items-center space-x-2 px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span>En línea</span>
              </div>
            </div>
          </div>
        </div>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.map((message) => (
            <div key={message.id} className={`flex ${message.isUser ? 'justify-end' : 'justify-start'}`}>
              <div className={`flex items-start space-x-3 max-w-3xl ${message.isUser ? 'flex-row-reverse space-x-reverse' : ''}`}>
                <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                  message.isUser ? 'bg-blue-500' : 'bg-gray-100'
                }`}>
                  {message.isUser ? 
                    <User className="w-4 h-4 text-white" /> : 
                    <Bot className="w-4 h-4 text-gray-600" />
                  }
                </div>
                <div className={`rounded-2xl p-4 ${
                  message.isUser 
                    ? 'bg-blue-500 text-white' 
                    : 'bg-gray-50 text-gray-900 border border-gray-200'
                }`}>
                  <div className={`text-sm ${message.isUser ? 'text-blue-100' : 'text-gray-900'}`}>
                    {formatMessageContent(message.content, message.type)}
                  </div>
                  <div className={`text-xs mt-2 ${message.isUser ? 'text-blue-200' : 'text-gray-400'}`}>
                    {message.timestamp.toLocaleTimeString('es-ES', { 
                      hour: '2-digit', 
                      minute: '2-digit' 
                    })}
                  </div>
                </div>
              </div>
            </div>
          ))}

          {isTyping && (
            <div className="flex justify-start">
              <div className="flex items-start space-x-3 max-w-3xl">
                <div className="w-8 h-8 bg-gray-100 rounded-full flex items-center justify-center">
                  <Bot className="w-4 h-4 text-gray-600" />
                </div>
                <div className="bg-gray-50 rounded-2xl p-4 border border-gray-200">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                  </div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Quick Suggestions */}
        <div className="px-6 py-4 border-t border-gray-100">
          <p className="text-sm text-gray-500 mb-3">Sugerencias rápidas:</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {quickSuggestions.map((suggestion, index) => (
              <button
                key={index}
                onClick={() => handleSendMessage(suggestion.text)}
                className="flex items-center space-x-2 p-3 text-left bg-gray-50 hover:bg-gray-100 rounded-lg border border-gray-200 hover:border-gray-300 transition-colors duration-200"
              >
                <suggestion.icon className="w-4 h-4 text-gray-500 flex-shrink-0" />
                <span className="text-sm text-gray-700 truncate">{suggestion.text}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Input Area */}
        <div className="p-6 border-t border-gray-100">
          <div className="flex space-x-4">
            <div className="flex-1 relative">
              <textarea
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Escribe tu pregunta sobre IA generativa..."
                className="w-full px-4 py-3 border border-gray-200 rounded-xl resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                rows={1}
              />
            </div>
            <button
              onClick={() => handleSendMessage()}
              disabled={!inputValue.trim() || isTyping}
              className="px-6 py-3 bg-blue-500 text-white rounded-xl hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors duration-200 flex items-center space-x-2"
            >
              <Send className="w-4 h-4" />
              <span className="hidden sm:inline">Enviar</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}