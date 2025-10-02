import React from 'react';
import { useIaChat } from '../context/iachatContext';

const ContextStatus = () => {
  const { context } = useIaChat();

  return (
    <div className="cart-status">
      <h2>Contexto</h2>
      <p>{JSON.stringify(context)}</p>
    </div>
  );
};

export default ContextStatus;