export const initialState = {
  memoryByContext: {
    general: [],
    example1: [],
    example2: [],
    example3: [],
    example4: []
  },
  scope: 'general'
};

export const ischatReducer = (state, action) => {

  switch (action.type) {
    case 'CHANGE_SCOPE': {
      return {
        ...state,
        scope: action.payload.newState
      };
    }
    case 'ADD_MEMORY_TO_SCOPE': {
      // console.log('ADDING MEMORY TO SCOPE', action.payload);
      const memories = Object.assign(initialState.memoryByContext);
      if (!memories[state.scope].includes(action.payload.memory)) memories[state.scope].push(action.payload.memory);

      return {
        ...state, memoryByContext: memories
      };
    }
    default:
      return initialState;
  }
};