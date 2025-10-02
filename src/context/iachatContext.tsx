import React, { createContext, useContext, useReducer } from 'react';
import { ischatReducer, initialState } from './iachatReducer';

// 1. Crear el contexto
const iachatContext = createContext();

// 2. Crear el Proveedor del contexto
export const IaChatProvider = ({ children }) => {
  const [state, dispatch] = useReducer(ischatReducer, initialState);

  // Funciones de utilidad para no tener que usar dispatch directamente en los componentes
  const changeScope = (newState) => {
    dispatch({ type: 'CHANGE_SCOPE', payload: { newState } });
  };

const addMemoryToScope = (memory) => {
    dispatch({ type: 'ADD_MEMORY_TO_SCOPE', payload: { memory } });
  };


  const value = {
    context: state,
    changeScope,
    addMemoryToScope
  };

  return (
    <iachatContext.Provider value={value}>
      {children}
    </iachatContext.Provider>
  );
};

export const useIaChat = () => {
  const context = useContext(iachatContext);
  if (context === undefined) {
    throw new Error('useIaChat must be used within a iachatProvider');
  }
  return context;
};