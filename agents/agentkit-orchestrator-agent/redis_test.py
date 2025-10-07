import asyncio
import json
from typing import Optional, Dict, List
from pydantic import BaseModel, Field

# Librería asíncrona para Redis
import redis.asyncio as aioredis 
from redis.asyncio import Redis 

# --- 1. Definición del Modelo de Sesión ---
# Usar Pydantic para validar y serializar/deserializar datos de forma eficiente
class AgentSession(BaseModel):
    user_id: str
    last_interaction: str = Field(default_factory=lambda: "No date")
    history: List[Dict[str, str]] = Field(default_factory=list)
    state: Dict = Field(default_factory=dict)

# --- 2. Clase de Gestor de Sesiones Asíncrono ---
class RedisSessionManager:
    """Gestiona la conexión y las operaciones CRUD asíncronas con Redis."""
    
    # TTL (Time To Live) de la sesión en segundos (ej: 1 hora)
    TTL_SECONDS = 3600
    
    def __init__(self, host: str, port: int, password: Optional[str] = None):
        self.host = host
        self.port = port
        self.password = password
        self._redis_pool: Optional[Redis] = None

    async def initialize(self):
        """Inicializa la conexión asíncrona (pool)."""
        if self._redis_pool is None:
            self._redis_pool = aioredis.Redis(
                host=self.host,
                port=self.port,
                password=self.password,
                decode_responses=True # Decodifica automáticamente a strings de Python
            )
            # Prueba de conexión asíncrona
            await self._redis_pool.ping()
            print("✅ Conexión asíncrona a Redis establecida.")

    async def close(self):
        """Cierra la conexión al finalizar la aplicación."""
        if self._redis_pool:
            await self._redis_pool.close()
            print("❌ Conexión a Redis cerrada.")

    async def save_session(self, session_id: str, session_data: AgentSession) -> bool:
        """Guarda una sesión con expiración (SETEX asíncrono)."""
        
        # Serializa el objeto Pydantic a una cadena JSON
        json_string = session_data.model_dump_json()
        
        # Operación SET con expiración (EX)
        result = await self._redis_pool.set(
            session_id, 
            json_string, 
            ex=self.TTL_SECONDS
        )
        return result

    async def load_session(self, session_id: str) -> Optional[AgentSession]:
        """Carga una sesión y la deserializa, o devuelve None si no existe/expiró."""
        
        # Operación GET asíncrona
        json_string = await self._redis_pool.get(session_id)
        
        if json_string:
            # Deserializa la cadena JSON de vuelta al modelo Pydantic
            data = json.loads(json_string)
            return AgentSession(**data)
        
        return None

    # --- MÉTODO AÑADIDO ---
    async def get_history_by_user_id(self, user_id: str) -> List[Dict[str, str]]:
        """
        Busca todas las sesiones de un usuario y devuelve su historial combinado.

        Este método escanea las claves que coinciden con el patrón de sesión,
        carga cada sesión y, si el user_id coincide, agrega su historial
        a una lista consolidada.
        """
        user_history = []
        # Escanea todas las claves que podrían ser sesiones de agente
        async for key in self._redis_pool.scan_iter("agent:session:*"):
            session_data = await self.load_session(key)
            # Comprueba si la sesión pertenece al usuario solicitado
            if session_data and session_data.user_id == user_id:
                user_history.extend(session_data.history)
        
        return user_history

# --- 3. Uso Principal del Código (Función Asíncrona) ---
async def main():
    # NOTA: En Docker Compose, el host es 'redis' y la contraseña la obtienes de .env
    manager = RedisSessionManager(
        host='localhost', # Cambiar a 'redis' si se ejecuta dentro de la red Docker
        port=6379,
        password='38UOs5TNhOm57TSr9Ww48SS7as5Kg2sm5LaAVKSXHHBWizmBv'
    )
    
    try:
        await manager.initialize()

        # Ejemplo de datos de sesión
        #session_id = "agent:session:user_45"
        #new_session = AgentSession(
        #    user_id="user_44",
        #    last_interaction="2025-08-07T14:35:00Z",
        #    history=[{"role": "user", "text": "Hola, necsdsdsdn resumen."}]
        #)

        # 1. Guardar la sesión
        #saved = await manager.save_session(session_id, new_session)
        #print(f"\nEstado de guardado: {saved}")
        
        # 2. Cargar la sesión
        #loaded_session = await manager.load_session("agent:session:user_42")
        
        #if loaded_session:
        #    print("\n--- Sesión Cargada ---")
        #    print(f"ID Usuario: {loaded_session.user_id}")
        #    print(f"Última Interacción: {loaded_session.last_interaction}")
        #    print(f"Historial: {loaded_session.history}")

        result = await manager.get_history_by_user_id("user_44")
        print(result)

    except Exception as e:
        print(f"\nOcurrió un error: {e}")
    finally:
        await manager.close()

if __name__ == "__main__":
    # Ejecuta la función principal asíncrona
    asyncio.run(main())


